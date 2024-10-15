from typing import Iterable, Callable

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from tqdm import tqdm

import random
from transformers import Dinov2Model

from ltvu.loss import get_losses_with_anchor
from ltvu.bbox_ops import bbox_xyhwToxyxy, generate_anchor_boxes_on_regions


base_sizes = torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    # 4 types of size
aspect_ratios = torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]


def detach_dict(d):
    # recursively detach all tensors in a dict
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = detach_dict(v)
        elif isinstance(v, torch.Tensor):
            d[k] = v.detach()
    return d


def build_backbone(backbone_name, backbone_type):
    if backbone_name == 'dinov2':
        assert backbone_type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('ignore', FutureWarning)
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(backbone_type))
        down_rate = 14
        if backbone_type == 'vitb14':
            backbone_dim = 768
        elif backbone_type == 'vits14':
            backbone_dim = 384
        elif backbone_type == 'vitl14':
            backbone_dim = 1024
        elif backbone_type == 'vitg14':
            backbone_dim = 1536
        else:
            raise NotImplementedError
    elif backbone_name == 'dinov2-hf':  # why not torch.hub? => just because I prefer huggingface
        backbone = Dinov2Model.from_pretrained('facebook/dinov2-base')
        down_rate = 14
        backbone_dim = 768
    else:
        raise NotImplementedError
    return backbone, down_rate, backbone_dim


def BasicBlock_Conv2D(in_dim, out_dim):
    module = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True))
    return module


class IntermediateFeatureExtractor:
    def __init__(self, model: nn.Module, layer_ids: Iterable[str]):
        self.model = model
        self.layer_ids = layer_ids
        self.features = {layer: torch.empty(0) for layer in layer_ids}

        for layer_id in layer_ids:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self.features[layer_id] = output
        return fn


class InferenceContext:
    def __init__(self, fp32_mm_precision, enable_autocast, autocast_dtype, enable_no_grad):
        self.fp32_mm_precision = fp32_mm_precision
        self.enable_no_grad = enable_no_grad
        self.enable_autocast = enable_autocast
        self.autocast_dtype = autocast_dtype
        self.prec_prev = None

    def __enter__(self):
        self.prec_prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.fp32_mm_precision)
        if self.enable_no_grad:
            self.no_grad = torch.no_grad()
            self.no_grad.__enter__()
        self.autocast = torch.autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.enable_autocast)
        self.autocast.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.autocast.__exit__(exc_type, exc_value, traceback)
        if self.enable_no_grad:
            self.no_grad.__exit__(exc_type, exc_value, traceback)
        torch.set_float32_matmul_precision(self.prec_prev)


class ClipMatcher(nn.Module):
    def __init__(self,
        compile_backbone = True,
        backbone_precision = 'bf16',
        backbone_fp32_mm_precision = 'medium',

        # model structure
        backbone_name = 'dinov2',
        backbone_type = 'vitb14',
        window_transformer = 5,  # sec
        resolution_transformer = 8,
        num_anchor_regions = 16,
        num_layers_st_transformer = 3,
        transformer_dropout = 0.,
        fix_backbone = True,

        # input size
        query_size = 448,
        clip_size_fine = 448,
        clip_size_coarse = 448,
        clip_num_frames = 30,

        # loss weights
        positive_threshold = .2,
        logit_scale = 1.,
        weight_bbox_center = 1.,
        weight_bbox_hw = 1.,
        weight_bbox_giou = .3,
        weight_prob = 100.,
        late_reduce = False,

        #### experiment-specific ####
        num_layers_cq_corr_transformer: int = 1,
        num_layers_stst_decoder: int = 0,
        t_short: int = 4,  # frames

        # cls token score
        enable_cls_token_score = False,

        # PCA Guide
        enable_pca_guide: bool = False,
        guide_from: str = 'backbone',
        guide_to_stx: bool = False,
        guide_to_stst: bool = False,
        weight_singular: float = 1e-6,
        weight_entropy: float = 1e-3,
        rank_pca: int = 4,

        debug = False,
        **kwargs
    ) -> None:
        super().__init__()

        # argument validation
        if enable_pca_guide:
            if guide_from == 'reduce':
                if late_reduce:
                    raise ValueError(f'PCA {guide_from=} from reduce layer with {late_reduce=} will be the same as guiding from backbone')
            if guide_to_stst:
                if num_layers_stst_decoder == 0:
                    raise ValueError('Cannot guide to STST without STST decoder')

        self.backbone, self.down_rate, self.backbone_dim = build_backbone(backbone_name, backbone_type)
        self.backbone_name = backbone_name
        prec = backbone_precision  # alias
        dtypes = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}
        self.backbone_dtype = dtypes[prec]
        self.backbone_autocast = prec != 'fp32'
        self.backbone_fp32_mm_precision = backbone_fp32_mm_precision

        self.query_size = query_size
        self.clip_size_fine = clip_size_fine
        self.clip_size_coarse = clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.window_transformer = window_transformer
        self.resolution_transformer = resolution_transformer
        self.num_anchor_regions = num_anchor_regions
        self.transformer_dropout = transformer_dropout
        self.fix_backbone = fix_backbone

        self.positive_threshold = positive_threshold
        self.logit_scale = logit_scale
        self.weight_bbox_center = weight_bbox_center
        self.weight_bbox_hw = weight_bbox_hw
        self.weight_bbox_giou = weight_bbox_giou
        self.weight_prob = weight_prob
        self.enable_cls_token_score = enable_cls_token_score
        self.late_reduce = late_reduce
        self.num_layers_cq_corr_transformer = num_layers_cq_corr_transformer
        self.num_layers_stst_decoder = num_layers_stst_decoder
        self.t_short = t_short

        self.enable_pca_guide = enable_pca_guide
        self.guide_from = guide_from
        self.guide_to_stx = guide_to_stx
        self.guide_to_stst = guide_to_stst

        self.weight_singular = weight_singular
        self.weight_entropy = weight_entropy
        self.rank_pca = rank_pca

        self.anchors_xyhw = generate_anchor_boxes_on_regions(
            image_size=[self.clip_size_coarse, self.clip_size_coarse],
            num_regions=[self.num_anchor_regions, self.num_anchor_regions])
        self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)  # non-trainable, [R^2*N*M,4]

        if fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # also done in the trainer
        layer_ids = []
        if enable_cls_token_score:
            # check layer names by printing `list(dict([*backbone.named_modules()]).keys())`
            layer_ids.append('encoder.layer.10.layer_scale2')
        # self.hidden_state_extractor = IntermediateFeatureExtractor(self.backbone, layer_ids)
        if compile_backbone:
            self.backbone = torch.compile(self.backbone)
            self.get_cross_cls_attn_score = torch.compile(self.get_cross_cls_attn_score)

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        # clip-query correspondence
        self.CQ_corr_transformer = []
        self.nhead = 4
        stx_in_dim = self.backbone_dim if self.late_reduce else 256

        for _ in range(self.num_layers_cq_corr_transformer):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=stx_in_dim,
                    nhead=self.nhead,
                    dim_feedforward=stx_in_dim*4,
                    dropout=transformer_dropout,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        # feature downsample layers
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(self.num_head_layers-1):
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        # spatial-temporal PE
        self.pe_3d = torch.zeros(1, clip_num_frames * self.resolution_transformer ** 2, 256)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        # spatial-temporal transformer layer
        self.feat_corr_transformer = []
        self.num_layers_st_transformer = num_layers_st_transformer
        for _ in range(self.num_layers_st_transformer):
            self.feat_corr_transformer.append(
                    torch.nn.TransformerEncoderLayer(
                        d_model=256,
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=transformer_dropout,
                        activation='gelu',
                        batch_first=True))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None


        if self.num_layers_stst_decoder > 0:
            self.pe_3d_st = torch.zeros(1, self.t_short * 32 ** 2, 256)
            self.pe_3d_st = nn.parameter.Parameter(self.pe_3d_st)
            self.stst_decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=self.nhead,
                    dim_feedforward=1024,
                    dropout=transformer_dropout,
                    activation='gelu',
                    batch_first=True),
                num_layers=self.num_layers_stst_decoder)

        # output head
        self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.num_anchor_regions)

        self.debug = debug

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def get_cross_cls_attn_score(self,latent_query, latent_clip):
        """
        latent_query: [b,1,c]
        latent_clip: [b*t,n,c]
        """
        BT, N, C = latent_clip.shape
        B = latent_query.shape[0]
        T = BT // B

        last_layer = self.backbone.encoder.layer[-1]
        Q_query = last_layer.attention.attention.query(latent_query)  # [b,1,c]
        K_clip = last_layer.attention.attention.key(latent_clip)  # [b*t,n,c]

        Q_query = repeat(Q_query, 'b 1 c -> (b t) 1 c', t=T)
        attn = torch.bmm(Q_query, K_clip.transpose(1, 2)) / C ** 0.5  # [b*t,1,n]
        attn = F.softmax(attn, dim=-1)  # [b*t,1,n]
        attn = repeat(attn, '(b t) 1 n2 -> (b t h) n1 n2', b=B, h=self.nhead, n1=N, n2=N)

        return attn

    def compute_pca_score_map(self, guide_feat):
        b = guide_feat.shape[0]
        guide_feat = rearrange(guide_feat, 'b c h w -> b (h w) c')  # [b,h2*w2,c]
        score_maps = []  # [b,h2*w2,H]
        for bidx in range(b):
            U, S, V = torch.pca_lowrank(guide_feat[bidx], q=1+self.nhead)  # [h2*w2,1+H], [1+H], [c,1+H]
            score_map = torch.matmul(guide_feat[bidx], V[:, 1:])  # [h2*w2,c] @ [c,H] -> [h2*w2,H]
            score_maps.append(score_map)
        score_maps = torch.stack(score_maps)  # [b,h2*w2,H]
        score_maps = 1. - torch.exp(-1 * score_maps ** 2 / 10)  # [b,h2*w2,H]
        return score_maps

    def extract_feature(self, x) -> dict:
        hidden_states = {}

        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            feat = self.backbone.get_intermediate_layers(x, n=1)[0]
            cls_token = feat[:, :1, :]  # [b,1,c]
            feat = feat[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = feat.shape[-1]
            feat = feat.reshape(b, h, w, dim).permute(0,3,1,2)

        elif self.backbone_name in ['dinov2', 'dinov2-hf']:
            b, _, h_origin, w_origin = x.shape
            if 'hf' in self.backbone_name:
                x_forward_outs = self.backbone.forward(x, output_hidden_states=True)
                feat = x_forward_outs.last_hidden_state
                hidden_states = x_forward_outs.hidden_states[-2]
                cls_token = rearrange(feat[:, :1, :], 'b 1 c -> b c 1')
                feat = feat[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
                h = int(h_origin / self.down_rate)
                w = int(w_origin / self.down_rate)
            else:
                feat, cls_token = self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)
                h = int(h_origin / self.backbone.patch_embed.patch_size[0])
                w = int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = feat.shape[-1]
            feat = feat.reshape(b, h, w, dim).permute(0,3,1,2)  # [b,c,h,w]

        else:
            raise NotImplementedError

        if torch.isnan(feat).any():
            raise ValueError('nan in feature')

        return {
            'feat': feat,
            'cls': cls_token,
            'h': h, 'w': w,
            'hidden_states': hidden_states
        }


    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            device = src.device
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw, device=device).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            # mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask

    def replicate_for_hnm(self, query_feat, clip_feat):
        '''
        query_feat in shape [b,c,h,w]
        clip_feat in shape [b*t,c,h,w]
        '''
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b

        clip_feat = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)

        new_clip_feat, new_query_feat = [], []
        for i in range(b):
            for j in range(b):
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[j])

        new_clip_feat = torch.stack(new_clip_feat)      # [b^2,t,c,h,w]
        new_query_feat = torch.stack(new_query_feat)    # [b^2,c,h,w]

        new_clip_feat = rearrange(new_clip_feat, 'b t c h w -> (b t) c h w')
        return new_clip_feat, new_query_feat

    def backbone_context(self):
        return InferenceContext(
            self.backbone_fp32_mm_precision,
            self.backbone_autocast,
            self.backbone_dtype,
            self.fix_backbone)


    def forward(
        self,
        segment,
        query,
        compute_loss = False,
        training = True,
        before_query_mask: None | torch.Tensor = None,
        gt_probs: None | torch.Tensor = None,
        gt_bboxes: None | torch.Tensor = None,  # yxyx
        use_hnm = False,
        rt_pos_queries = None,
        rt_pos = False,
        sim_mode = 'max',
        sim_thr = 0.0,
        enable_rt_pq_threshold=False,
        **kwargs
    ):
        '''
        clip: in shape [b,t,c,h,w]
        query: in shape [b,c,h2,w2]
        before_query_mask:
        gt_bboxes:
        '''
        b, t = segment.shape[:2]
        device = segment.device

        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        with self.backbone_context():
            clip_feat_dict = self.extract_feature(segment)
            if rt_pos and random.randint(0, 1) == 1:
                rt_pos_queries = rearrange(rt_pos_queries, 'b t c h w -> (b t) c h w') # [b*t,c,h,w]
                rt_pos_queries_feat_dict = self.extract_feature(rt_pos_queries)
                query_feat_dict = self.extract_feature(query)
                rt_pos_queries_cls, query_cls = rt_pos_queries_feat_dict['cls'], query_feat_dict['cls']
                query_cls = rearrange(query_cls, 'b c 1 -> b 1 c').expand(-1, t, -1) # [b,t,c]
                rt_pos_queries_cls = rearrange(rt_pos_queries_cls.squeeze(-1), '(b t) c -> b t c', b= b, t=t) # [b,t,c]

                sim = F.cosine_similarity(rt_pos_queries_cls, query_cls, dim=-1) # [b,t]

                sim_mask = sim > sim_thr # [b,t]
                sim_mask_num = sim_mask.sum() / b
                batch_has_valid = sim_mask.any(dim=-1) # [b]

                rand_indices_per_batch = torch.zeros(b, dtype=torch.long, device=sim.device)  # [b]
                if batch_has_valid.any():
                    valid_sim_mask = sim_mask.float()  # [b, t]
                    rand_indices_per_batch[batch_has_valid] = torch.multinomial(valid_sim_mask[batch_has_valid], 1).squeeze(1)  # [b]

                if sim_mode == 'max':
                    top_sim_idx = sim.argmax(dim=-1) # [b,1]
                elif sim_mode == 'min':
                    top_sim_idx = sim.argmin(dim=-1, keepdim=True)  # [b, 1]

                final_top_sim_idx = torch.where(batch_has_valid, rand_indices_per_batch, top_sim_idx)  # [b, 1]
                rt_pos_queries = rearrange(rt_pos_queries, '(b t) c h w -> b t c h w', b=b, t=t) # [b,t,c,h,w]

                if enable_rt_pq_threshold:
                    query = rt_pos_queries[torch.arange(b), final_top_sim_idx]  # [b, c, h2, w2]
                else:
                    query = rt_pos_queries[torch.arange(b), top_sim_idx] # [b,c,h2,w2]
            query_feat_dict = self.extract_feature(query)



        query_feat = query_feat_dict['feat']
        clip_feat = clip_feat_dict['feat']
        h, w = clip_feat_dict['h'], clip_feat_dict['w']

        if self.enable_pca_guide and self.guide_from == 'backbone':
            score_maps = self.compute_pca_score_map(query_feat)  # [b,h2*w2,H]

        # reduce channel size
        if not self.late_reduce:
            all_feat = torch.cat([query_feat, clip_feat], dim=0)
            all_feat = self.reduce(all_feat)
            query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if self.enable_pca_guide and self.guide_from == 'reduce':
            score_maps = self.compute_pca_score_map(query_feat)  # [b,h2*w2,H]

        if use_hnm and compute_loss:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   # b -> b^2
            b **= 2

        # masks
        nc, ts = t // self.t_short, self.t_short
        stx_tgt_mask = None   # [b*t*H,h*w,h*w], Q, K
        stx_mem_mask = None   # [b*t*H,h1*w1,h2*w2]
        stst_mem_mask = None  # [b*H,ts*h1*w1,h2*w2]  # h1*w1 for clip(Q), h2*w2 for query(K)

        # cls token score
        if self.enable_cls_token_score:
            latent_query = query_feat_dict['hidden_states']
            latent_clip = clip_feat_dict['hidden_states']
            latent_query_cls = latent_query[:, :1]  # [b,1,c]
            latent_clip_non_cls = latent_clip[:, 1:]  # [b*t,n,c]
            with self.backbone_context():
                cls_mask = self.get_cross_cls_attn_score(latent_query_cls, latent_clip_non_cls)
            stx_tgt_mask = cls_mask

        # pca guide
        if self.enable_pca_guide:
            if self.guide_to_stx:
                raise NotImplementedError
            if self.guide_to_stst:
                pca_mask = repeat(
                    score_maps, 'b (h2 w2) H -> (b H) (ts h1 w1) (h2 w2)', ts=ts, h1=h, w1=w, h2=h, w2=w)
                stst_mem_mask = pca_mask

        # find spatial correspondence between query-frame
        query_feat_expanded = repeat(query_feat, 'b c h w -> (b t) (h w) c', t=t)  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')            # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            layer: nn.TransformerDecoderLayer  # written for pylance
            clip_feat = layer.forward(
                tgt=clip_feat, memory=query_feat_expanded,
                tgt_mask=stx_tgt_mask, memory_mask=stx_mem_mask)
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)  # [b*t,c,h,w]

        if self.late_reduce:
            all_feat = torch.cat([query_feat, clip_feat], dim=0)
            all_feat = self.reduce(all_feat)
            query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if self.num_layers_stst_decoder > 0:
            clip_feat = rearrange(
                clip_feat, '(b nc ts) c h w -> nc b (ts h w) c', b=b, ts=ts, nc=nc, h=h, w=w)
            clip_feat = clip_feat + self.pe_3d_st
            query_feat = rearrange(query_feat, 'b c h w -> b (h w) c', b=b, h=h)
            _feats = []
            for cidx in range(nc):  # for saving VRAM or memory mask being too large
                _feat = self.stst_decoder.forward(clip_feat[cidx], query_feat, memory_mask=stst_mem_mask)
                _feats.append(_feat)
            clip_feat = torch.stack(_feats)  # [nc,b,ts*h*w,c]
            clip_feat = rearrange(
                clip_feat, 'nc b (ts h w) c -> (b nc ts) c h w', b=b, ts=ts, nc=nc, h=h, w=w)

        # down-size features and find spatial-temporal correspondence
        for down_head in self.down_heads:
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(
                    clip_feat, 'b (t h w) c -> (b t) c h w',
                    b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break  # ? why break here
            clip_feat = down_head.forward(clip_feat)

        # refine anchors
        anchors_xyhw = self.anchors_xyhw.to(device)                             # [N,4]
        anchors_xyxy = self.anchors_xyxy.to(device)                             # [N,4]
        anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           # [1,1,N,4]

        bbox_refine, prob = self.head.forward(clip_feat)                        # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        prob = self.logit_scale*rearrange(prob, '(b t) N c -> b t N c', b=b, t=t) # [b,t,N,1]
        prob = prob.squeeze(-1)                                                 # [b,t,N]
        bbox_refine += anchors_xyhw                                             # [b,t,N,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        pred_dict = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob,               # [b,t,N], logits
            'anchor': anchors_xyxy      # [1,1,N,4]
        }

        output_dict = {}
        if compute_loss:
            assert before_query_mask is not None
            assert gt_probs is not None
            assert gt_bboxes is not None

            # rename variables for `get_losses_with_anchor` interface
            gts = {
                'before_query': before_query_mask,  # [b,t]
                'clip_with_bbox': gt_probs,         # [b,t]
                'clip_bbox': gt_bboxes,             # [b,t,4]
                # 'hw': None,                     # [b,t,2]
                # 'center': None,                 # [b,t,2]
            }
            # acutal loss calculation
            loss_dict, preds_top, gts, pos_mask = get_losses_with_anchor(
                pred_dict, gts,
                training=training,
                positive_threshold=self.positive_threshold,
                weight_bbox_center=self.weight_bbox_center,
                weight_bbox_hw=self.weight_bbox_hw,
                weight_bbox_giou=self.weight_bbox_giou,
                weight_prob=self.weight_prob,
            )

            loss_names = [k.replace('loss_', '') for k in loss_dict.keys() if 'loss_' in k]
            total_loss: torch.Tensor = torch.tensor(0., dtype=torch.float32, device=device, requires_grad=True)
            for loss_name in loss_names:
                w, l = loss_dict[f'weight_{loss_name}'], loss_dict[f'loss_{loss_name}']
                if training:
                    assert l.requires_grad, f'{loss_name} should require grad'
                total_loss = total_loss + w * l


            if self.weight_singular > 0 or self.weight_entropy > 0:
                # penalize entropy of a normed score map while limiting sigma's
                # entropy -> activate only specific parts
                # sigma -> limit the overall score magnitude
                loss_singular = torch.tensor(0., dtype=query_feat.dtype, device=device)
                loss_entropy = torch.tensor(0., dtype=query_feat.dtype, device=device)
                for bidx in range(b):
                    U, S, V = torch.pca_lowrank(query_feat[bidx], q=self.rank_pca)  # [h*w,Q], [Q], [c,Q]
                    query_feat_proj = torch.matmul(query_feat[bidx], V).abs()  # [h*w,Q]
                    tmax = query_feat_proj.max(dim=0, keepdim=True)[0]  # [1,Q]
                    query_feat_proj = query_feat_proj / tmax  # values in [-1,1]
                    query_feat_proj = F.softmax(query_feat_proj, dim=0)  # each map sums to 1   # [h*w,Q]
                    query_feat_proj = query_feat_proj.clamp(1e-6, 1-1e-6)  # avoid log(0)
                    self_entropy = -(query_feat_proj * query_feat_proj.log()).sum(dim=0).mean()
                    loss_entropy = loss_entropy + self_entropy.mean()
                    max_sigma = torch.tensor(1000., dtype=S.dtype, device=device)
                    loss_singular = loss_singular + -torch.minimum(S[..., 1:], max_sigma).sum()
                loss_singular = loss_singular / b
                loss_entropy = loss_entropy / b
                if self.weight_singular > 0:
                    total_loss = total_loss + self.weight_singular * loss_singular
                if self.weight_entropy > 0:
                    total_loss = total_loss + self.weight_entropy * loss_entropy
                # print(loss_singular, loss_entropy, loss_dict['loss_prob'])

            if self.debug:
                tqdm.write(repr(loss_dict['loss_prob']))

            # for logging - losses
            loss_dict = detach_dict(loss_dict)
            log_dict = {
                'loss': total_loss.detach(),
                'loss_bbox_center': loss_dict['loss_bbox_center'].mean(),
                'loss_bbox_hw': loss_dict['loss_bbox_hw'].mean(),
                'loss_bbox_giou': loss_dict['loss_bbox_giou'].mean(),
                'loss_prob': loss_dict['loss_prob'].mean(),
            }

            if self.weight_singular > 0:
                log_dict.update({'loss_singular': loss_singular.detach()})
            if self.weight_entropy > 0:
                log_dict.update({'loss_entropy': loss_entropy.detach()})


            # for logging - metrics
            preds_top = detach_dict(preds_top)
            prob: torch.Tensor = prob.detach()
            b, t, N = prob.shape
            ious, gious = loss_dict['iou'], loss_dict['giou']  # both [b*t*N]
            prob_theta = .5
            if training:  # measure for all positive anchors
                if pos_mask.sum() > 0:
                    ious, gious = ious[pos_mask], gious[pos_mask]  # both [#pos_anchors]
                    prob = prob.flatten()[pos_mask]  # [b*t*N] -> [#pos_anchors]
                    prob_accuracy = (prob.sigmoid() > prob_theta).float().mean()
                else:
                    prob_accuracy = torch.tensor(0., dtype=torch.float32, device=device)
            else:  # measure for per-frame top-1 anchors
                pos_mask = gt_probs.bool().flatten()  # [b,t]
                top_idx = rearrange(prob.argmax(dim=-1, keepdim=True), 'b t 1 -> (b t) 1')  # [b*t, 1]
                ious, gious = rearrange(ious, '(b t N) -> (b t) N', b=b, t=t), rearrange(gious, '(b t N) -> (b t) N', b=b, t=t)
                ious, gious = torch.take_along_dim(ious, top_idx, dim=-1), torch.take_along_dim(gious, top_idx, dim=-1)  # [b*t]
                ious, gious = ious[pos_mask], gious[pos_mask]
                prob = preds_top['prob']  # [b,t]
                prob_accuracy = (prob.sigmoid() > prob_theta) == gt_probs.bool()  # [b,t]
                prob_accuracy = prob_accuracy.float().mean()
            ious, gious = ious.mean(), gious.mean()
            log_dict.update({
                'iou': ious,
                'giou': gious,
                'prob_acc': prob_accuracy,
            })
            if locals().get('sim_mask_num') is not None and enable_rt_pq_threshold:
                log_dict.update({'sim_mask_num': locals().get('sim_mask_num').item()})

            # not for logging but just in case we need it
            info_dict = {
                'loss_dict': loss_dict,     # losses starting with 'loss_', weights starting with 'weight_', iou, giou
                'preds_top': preds_top,     # bbox: [b,t,4], prob: [b,t]
                'gts': gts,                 # gts with hw, center computed
            }

            # gather all outputs
            output_dict.update({'loss': total_loss})  # for backward
            output_dict.update({'log_dict': log_dict})  # for logging
            output_dict.update({'info_dict': info_dict})  # for debugging
        output_dict.update({'pred_dict': detach_dict(pred_dict)})

        return output_dict


class Head(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3

        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B,c,h=8,w=8]
        '''
        if self.num_up_layers > 0:
            x = self.up_convs(x)     # [B,c,h=16,w=16]

        B, c, h, w = x.shape

        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w]
        # dpout pos 1, seems better
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,h,w]
        feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,h,w]

        out_reg = self.regression_head(feat_reg)
        out_cls = self.classification_head(feat_cls)

        out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

        return out_reg, out_cls


if __name__ == '__main__':
    import hydra
    from omegaconf import OmegaConf, DictConfig

    @hydra.main(config_path='../../config', config_name='base', version_base='1.3')
    def main(config: DictConfig):
        model = ClipMatcher(config).cuda()
        print('Model with {} parameters'.format(sum(p.numel() for p in model.parameters())))
        print(config.model.cpt_path)
        checkpoint = torch.load(config.model.cpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        torch.set_grad_enabled(False)
        del checkpoint



    main()
