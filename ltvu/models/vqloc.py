from typing import Iterable, Callable

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math

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

        # experiment-specific
        enable_cls_token_score = False,
        enable_pca_guide = False,
        pca_guide_version: int|float = 1,
        pca_guide_ablation = False,
        weight_singular = 1e-6,
        weight_pca_recon = .1,
        weight_entropy = 1e-3,
        num_layers_cq_corr_transformer=1,

        debug = False,
        **kwargs
    ) -> None:
        super().__init__()

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

        self.enable_pca_guide = enable_pca_guide
        self.pca_guide_version = pca_guide_version
        self.pca_guide_ablation = pca_guide_ablation
        self.weight_singular = weight_singular
        self.weight_pca_recon = weight_pca_recon
        self.weight_entropy = weight_entropy

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
        num_head_layers = int(math.log2(self.clip_feat_size_coarse // self.resolution_transformer))
        self.down_heads = []
        # self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(num_head_layers):
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


        if self.enable_pca_guide:
            if self.pca_guide_version == 2:
                self.unreduce = nn.Linear(256, 768)

            if self.pca_guide_version in [5, 6, 61]:
                # assert num_layers_st_transformer == 0, 'num_layers_st_transformer should be 0'
                # assert resolution_transformer == 32, 'resolution_transformer should be 32'
                # assert num_anchor_regions == 32, 'num_anchor_regions should be 32'
                self.t_short = 4
                self.pe_3d_dense = nn.parameter.Parameter(torch.zeros(1, self.t_short * 32 ** 2, 256))
                self.st_decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        d_model=256,
                        nhead=self.nhead,
                        dim_feedforward=1024,
                        dropout=transformer_dropout,
                        activation='gelu',
                        batch_first=True),
                    num_layers=1)

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
                feat = self.backbone.get_intermediate_layers(x, n=1)[0]
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
        layer_id = 'encoder.layer.10.layer_scale2'  # the second last layer
        with self.backbone_context():
            clip_feat_dict = self.extract_feature(segment)
            if rt_pos and random.randint(0, 1) == 1:
                rt_pos_queries = rearrange(rt_pos_queries, 'b t c h w -> (b t) c h w') # [b*t,c,h,w]
                rt_pos_queries_feat_dict = self.extract_feature(rt_pos_queries)
                rt_pos_queries_cls = rearrange(rt_pos_queries_cls.squeeze(-1), '(b t) c -> b t c', b= b, t=t) # [b,t,c]
                query_feat_dict = self.extract_feature(query)
                rt_pos_queries_cls, query_cls = rt_pos_queries_feat_dict['cls'], query_feat_dict['cls']
                query_cls = rearrange(query_cls, 'b c 1 -> b 1 c').expand(-1, t, -1) # [b,t,c]

                sim = F.cosine_similarity(rt_pos_queries_cls, query_cls, dim=-1) # [b,t]

                if sim_mode == 'max':
                    _, top_sim_idx = sim.topk(1, dim=-1) # [b,1]
                elif sim_mode == 'min':
                    top_sim_idx = sim.argmin(dim=-1, keepdim=True)  # [b, 1]

                rt_pos_queries = rearrange(rt_pos_queries, '(b t) c h w -> b t c h w', b=b, t=t) # [b,t,c,h,w]

                query = rt_pos_queries[torch.arange(b), top_sim_idx.squeeze(1)] # [b,c,h2,w2]
            query_feat_dict = self.extract_feature(query)




        query_feat = query_dino_feat = query_feat_dict['feat']
        clip_feat = clip_feat_dict['feat']
        h, w = clip_feat_dict['h'], clip_feat_dict['w']

        # reduce channel size
        if not self.late_reduce:
            all_feat = torch.cat([query_feat, clip_feat], dim=0)
            all_feat = self.reduce(all_feat)
            query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if use_hnm and compute_loss:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   # b -> b^2
            b **= 2


        # cls_mask
        cls_mask_sa = None  # [b*t*H,h*w,h*w], Q, K
        if self.enable_cls_token_score:
            latent_query = query_feat_dict['hidden_states']
            latent_clip = clip_feat_dict['hidden_states']
            latent_query_cls = latent_query[:, :1]  # [b,1,c]
            latent_clip_non_cls = latent_clip[:, 1:]  # [b*t,n,c]
            with self.backbone_context():
                cls_mask_sa = self.get_cross_cls_attn_score(latent_query_cls, latent_clip_non_cls)

        memory_mask_ca = None  # [b*t*H,h1*w1,h2*w2]  # h1*w1 for clip(Q), h2*w2 for query(K)
        if self.enable_pca_guide:
            if self.pca_guide_version in [5, 6, 61]:
                _query_dino_feat = rearrange(query_dino_feat, 'b c h w -> b (h w) c')  # [b,h2*w2,c]
                score_maps = []  # [b,h2*w2,H]
                for bidx in range(b):
                    U, S, V = torch.pca_lowrank(_query_dino_feat[bidx], q=1+self.nhead)  # [h2*w2,1+H], [1+H], [c,1+H]
                    score_map = torch.matmul(_query_dino_feat[bidx], V[:, 1:])  # [h2*w2,c] @ [c,H] -> [h2*w2,H]
                    if self.pca_guide_version == 6:
                        score_map = 1. - torch.exp(-1 * score_map ** 2 / 10)  # [h2*w2,H]
                    elif self.pca_guide_version == 61:
                        pass  # noop
                    score_maps.append(score_map)
                score_maps = torch.stack(score_maps)  # [b,h2*w2,H]

                if self.pca_guide_version == 5:
                    memory_mask_ca = repeat(
                        score_maps, 'b (h2 w2) H -> (b t H) (h1 w1) (h2 w2)',
                        t=t, h1=h, w1=w, h2=h, w2=w)


        # find spatial correspondence between query-frame
        query_feat = repeat(query_feat, 'b c h w -> (b t) (h w) c', t=t)  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')            # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            layer: nn.TransformerDecoderLayer  # written for pylance
            clip_feat = layer.forward(
                tgt=clip_feat, memory=query_feat,
                tgt_mask=cls_mask_sa, memory_mask=memory_mask_ca)                        # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)  # [b*t,c,h,w]

        if self.late_reduce:
            clip_feat = self.reduce(clip_feat)

        if self.enable_pca_guide:
            if self.pca_guide_version in [6, 61, 62]:
                nc, ts = t // self.t_short, self.t_short
                memory_mask_ca = repeat(
                    score_maps, 'b (h2 w2) H -> (b H) (ts h1 w1) (h2 w2)', ts=ts, h1=h, w1=w, h2=h, w2=w)
                if self.pca_guide_ablation:
                    memory_mask_ca = None
                clip_feat = rearrange(
                    clip_feat, '(b nc ts) c h w -> nc b (ts h w) c', b=b, ts=ts, nc=nc, h=h, w=w)
                clip_feat = clip_feat + self.pe_3d_dense
                _feats = []
                for cidx in range(nc):  # for saving VRAM or memory mask being too large
                    _query_dino_feat = rearrange(
                        query_feat, '(b t) (h w) c -> t b (h w) c', b=b, t=t, h=h)[0]  # collapse t dimension
                    _feat = self.st_decoder.forward(
                        clip_feat[cidx], _query_dino_feat, memory_mask=memory_mask_ca)
                    _feats.append(_feat)
                clip_feat = torch.stack(_feats)  # [nc,b,ts*h*w,c]
                clip_feat = rearrange(
                    clip_feat, 'nc b (ts h w) c -> (b nc ts) c h w', b=b, ts=ts, nc=nc, h=h, w=w)

        # down-size features and find spatial-temporal correspondence
        for head in self.down_heads:
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(
                    clip_feat, 'b (t h w) c -> (b t) c h w',
                    b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break  # ? why break here
            clip_feat = head(clip_feat)

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


            if self.enable_pca_guide:
                Q = 4
                _query_feat = rearrange(query_feat, '(b t) (h w) c -> t (b h w) c', b=b, t=t, h=h)[0]  # collapse t dimension
                _query_dino_feat = rearrange(query_dino_feat, 'b c h w -> (b h w) c', b=b)  # [b*h*w,c]

                if self.pca_guide_version == 1:
                    # simply forcing to extract query-wise features by maximizing 2 ~ Q-th singular values
                    U, S, V = torch.pca_lowrank(_query_feat, q=Q)
                    query_feat_proj = torch.matmul(_query_feat, V)  # [b*h*w,Q]
                    query_feat_proj = rearrange(query_feat_proj, '(b h w) q -> b (h w) q', b=b, h=h)

                    max_sigma = torch.tensor(1000., dtype=S.dtype, device=device)
                    loss_singular = -torch.minimum(S, max_sigma)[..., 1:].sum(dim=-1)
                    loss_singular = loss_singular.mean()
                    total_loss = total_loss + self.weight_singular * loss_singular
                    # print(loss_singular, loss_dict['loss_prob'])

                elif self.pca_guide_version == 2:
                    # forcing to reconstruct the DINO features by PCA
                    _query_feat = self.unreduce(_query_feat)  # [b*h*w,768]
                    _query_feat = rearrange(_query_feat, '(b h w) c -> b (h w) c', b=b, h=h)
                    latent_query = query_feat_dict['hidden_states'].detach()  # [b,1+h*w,c]
                    latent_query = latent_query[:, 1:]  # [b,h*w,c]
                    query_feats = torch.cat([latent_query, _query_feat], dim=1)  # [b,2*h*w,c]
                    loss_pca_recon = torch.tensor(0., dtype=_query_feat.dtype, device=device)
                    for bidx in range(b):
                        U, S, V = torch.pca_lowrank(query_feats[bidx], q=Q)
                        query_feats_proj = torch.matmul(query_feats[bidx], V)  # [2*h*w,Q]
                        hw = query_feats_proj.shape[0] // 2
                        latent_query_proj, query_feat_proj = query_feats_proj[:hw], query_feats_proj[hw:]  # [h*w,Q]
                        loss_pca_recon = loss_pca_recon + (latent_query_proj[..., 1:] - query_feat_proj[..., 1:]).norm(dim=-1).sigmoid().mean()
                    loss_pca_recon = loss_pca_recon / b
                    total_loss = total_loss + self.weight_pca_recon * loss_pca_recon

                elif self.pca_guide_version == 3:
                    # penalize entropy of a normed score map while limiting sigma's
                    # entropy -> activate only specific parts
                    # sigma -> limit the overall score magnitude
                    _query_feat = rearrange(_query_feat, '(b h w) c -> b (h w) c', b=b, h=h)
                    loss_singular = torch.tensor(0., dtype=_query_feat.dtype, device=device)
                    loss_entropy = torch.tensor(0., dtype=_query_feat.dtype, device=device)
                    for bidx in range(b):
                        U, S, V = torch.pca_lowrank(_query_feat[bidx], q=Q)  # [h*w,Q], [Q], [c,Q]
                        query_feat_proj = torch.matmul(_query_feat[bidx], V).abs()  # [h*w,Q]
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
                    total_loss = total_loss + self.weight_singular * loss_singular + self.weight_entropy * loss_entropy
                    # print(loss_singular, loss_entropy, loss_dict['loss_prob'])

                elif self.pca_guide_version == 4:
                    # hungarian-matches between query and GT crops
                    pass

                elif self.pca_guide_version == 5:  # checked
                    # DINO PCA score map as a spatial attention guide
                    # print(loss_dict['loss_prob'])
                    pass

                elif self.pca_guide_version == 6:  # checked
                    # DINO PCA score map as a spatio-temporal attention guide
                    pass

                elif self.pca_guide_version == 61:
                    # DINO feature -> itself
                    pass
                elif self.pca_guide_version == 62:
                    # DINO feature -> simple linear (as PCA is a linear transformation)
                    pass
                elif self.pca_guide_version == 63:
                    # DINO feature -> simple unet
                    pass

                # from io import BytesIO
                # from PIL import Image
                # from imgcat import imgcat
                # import seaborn as sns
                # import matplotlib.pyplot as plt
                # from diffusers.utils import make_image_grid
                # images = []
                # plot_io = BytesIO()
                # sns.heatmap((s:=query_feat_proj[0, 0, ..., 0].cpu().detach()).numpy(), square=True, cbar=True, cmap='PiYG', vmax=s.abs().max(), vmin=-s.abs().max())
                # plt.savefig(plot_io, format='png')
                # plt.close()
                # image = Image.open(plot_io)
                # images.append(image)
                # query_ = rearrange(query, 'b c h w -> b h w c')
                # query_ = (query_ - query_.min()) / (query_.max() - query_.min()) * 255
                # image = Image.fromarray(query_[0].cpu().detach().numpy().astype('uint8'))
                # print('hi')
                # images.append(image)
                # image = make_image_grid(images, rows=1, cols=len(images), resize=448)
                # imgcat(image)

                # print(self.weight_pca * loss_pca)
            if self.debug:
                print(loss_dict['loss_prob'])


            # for logging - losses
            loss_dict = detach_dict(loss_dict)
            log_dict = {
                'loss': total_loss.detach(),
                'loss_bbox_center': loss_dict['loss_bbox_center'].mean(),
                'loss_bbox_hw': loss_dict['loss_bbox_hw'].mean(),
                'loss_bbox_giou': loss_dict['loss_bbox_giou'].mean(),
                'loss_prob': loss_dict['loss_prob'].mean(),
            }

            if self.enable_pca_guide:
                if self.pca_guide_version == 1:
                    log_dict.update({'loss_singular': loss_singular.detach()})

                elif self.pca_guide_version == 2:
                    log_dict.update({'loss_pca_recon': loss_pca_recon.detach()})

                elif self.pca_guide_version == 3:
                    log_dict.update({
                        'loss_singular': loss_singular.detach(),
                        'loss_entropy': loss_entropy.detach()
                    })


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
