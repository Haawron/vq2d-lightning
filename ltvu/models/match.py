import itertools
from typing import Iterable, Callable
from collections import OrderedDict

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from tqdm import tqdm

from transformers import Dinov2Model
from transformers import CLIPModel

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
        down_rate = 14
        if backbone_type == 'vitb14':
            backbone_path = 'facebook/dinov2-base'
            backbone_dim = 768
        elif backbone_type == 'vits14':
            backbone_path = 'facebook/dinov2-small'
            backbone_dim = 384
        elif backbone_type == 'vitl14':
            backbone_path = 'facebook/dinov2-large'
            backbone_dim = 1024
        elif backbone_type == 'vitg14':
            backbone_path = 'facebook/dinov2-giant'
            backbone_dim = 1536
        backbone = Dinov2Model.from_pretrained(backbone_path)
    elif backbone_name == 'clip-hf':
        if backbone_type == 'vitb16':
            backbone_path = 'openai/clip-vit-base-patch16'
            backbone_dim = 512
            down_rate = 16
        elif backbone_type == 'vitl14':
            backbone_path = 'openai/clip-vit-large-patch14-336'
            backbone_dim = 1024
            down_rate = 14
        backbone = CLIPModel.from_pretrained(backbone_path).vision_model
    else:
        raise NotImplementedError
    return backbone, down_rate, backbone_dim


def BasicBlock_Conv2D(in_dim, out_dim):
    module = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True))
    return module


# https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py
class TemporalShift(nn.Module):
    def __init__(self, net, num_frames=32, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.num_frames = num_frames
        self.fold_div = n_div

    def forward(self, tgt, *args, **kwargs):
        tgt = self.shift(tgt, self.num_frames, fold_div=self.fold_div)
        return self.net(tgt, *args, **kwargs)

    @staticmethod
    def shift(x, t, fold_div=8):
        nt, hw, c = x.size()
        h, w = int(hw ** 0.5), int(hw ** 0.5)
        b = nt // t
        x = rearrange(x, '(b t) (h w) c -> b t c h w', b=b, t=t, h=h, w=w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return rearrange(out, 'b t c h w -> (b t) (h w) c')


class TemporalShiftConv(TemporalShift):
    def forward(self, x):
        bt, c, h, w = x.shape
        x = rearrange(x, '(b t) c h w -> (b t) (h w) c', t=self.num_frames)
        x = self.shift(x, self.num_frames, fold_div=self.fold_div)
        x = rearrange(x, '(b t) (h w) c -> (b t) c h w', t=self.num_frames, h=h)
        return self.net(x)


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


class MATCH(nn.Module):
    def __init__(self,
        compile_backbone = True,
        backbone_precision = 'bf16',
        backbone_fp32_mm_precision = 'medium',

        # model structure
        backbone_name = 'dinov2',
        backbone_type = 'vitb14',
        resolution_transformer = 8,
        num_anchor_regions = 16,
        transformer_dropout = 0.,
        fix_backbone = True,
        base_sizes: torch.Tensor = base_sizes,

        # input size
        query_size = 448,
        clip_size_fine = 448,
        clip_size_coarse = 448,
        clip_num_frames = 32,

        # loss weights
        positive_threshold = .2,
        logit_scale = 1.,
        weight_bbox_center = 1.,
        weight_bbox_hw = 1.,
        weight_bbox_giou = .3,
        weight_prob = 100.,

        #### experiment-specific ####
        num_layers_spatial_decoder: int = 1,
        nhead_stx: int = 4,

        # global guidance
        enable_global_guidance = False,
        cls_repair_neighbor = False,

        # Local guidance
        enable_local_guidance: bool = False,
        weight_entropy: float = 0.,
        rank_pca: int = 4,

        # temporal modeling
        enable_temporal_shift_conv_summary: bool = False,

        conv_summary_layers: int = 0,

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

        # self.window_transformer = window_transformer
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

        self.enable_global_guidance = enable_global_guidance
        self.cls_repair_neighbor = cls_repair_neighbor
        self.num_layers_spatial_decoder = num_layers_spatial_decoder

        self.enable_local_guidance = enable_local_guidance

        self.weight_entropy = weight_entropy
        self.rank_pca = rank_pca

        self.enable_temporal_shift_conv_summary = enable_temporal_shift_conv_summary

        self.conv_summary_layers = conv_summary_layers

        if fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # also done in the trainer

        # spatial decoder
        self.nhead = nhead_stx
        self.spatial_decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
                d_model=self.backbone_dim,
                nhead=self.nhead,
                dim_feedforward=self.backbone_dim*4,
                dropout=transformer_dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=self.num_layers_spatial_decoder
        )

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        if self.conv_summary_layers > 0:
            def conv_block(in_dim, out_dim):
                return nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_dim, out_dim, 3, padding=1)),
                    ('bn', nn.BatchNorm2d(out_dim)),
                    ('relu', nn.LeakyReLU(inplace=True))
                ]))
            self.conv_summary = nn.ModuleList([conv_block(256, 256) for _ in range(self.conv_summary_layers)])
        else:
            self.conv_summary = None

        if self.enable_temporal_shift_conv_summary:
            assert self.conv_summary is not None, 'conv_summary must be enabled to use temporal shift'
            # shift only the first layer
            self.conv_summary[0].conv = TemporalShiftConv(self.conv_summary[0].conv, num_frames=clip_num_frames, n_div=8)

        # anchors
        if not isinstance(base_sizes, torch.Tensor):
            if isinstance(base_sizes[0], list):
                base_sizes = torch.tensor(base_sizes, dtype=torch.float32)
            elif isinstance(base_sizes[0], int):
                base_sizes = torch.tensor([base_sizes] * 2, dtype=torch.float32).T
        self.anchors_yxhw = generate_anchor_boxes_on_regions(
            image_size=[self.clip_size_coarse, self.clip_size_coarse],
            num_regions=[self.num_anchor_regions, self.num_anchor_regions],
            base_sizes=base_sizes)
        self.anchors_yxhw = self.anchors_yxhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_yxyx = bbox_xyhwToxyxy(self.anchors_yxhw)  # non-trainable, [R^2*N*M,4]

        # output head
        self.head = Head(
            in_dim=256, in_res=self.resolution_transformer, out_res=self.num_anchor_regions,
            n=len(base_sizes)
        )

        self.debug = debug

        if compile_backbone:
            self.backbone = torch.compile(self.backbone)
            self.spatial_decoder = torch.compile(self.spatial_decoder)
            self.reduce = torch.compile(self.reduce)
            self.head = torch.compile(self.head)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def get_cross_cls_attn_score(self, latent_query, latent_clip):
        """
        latent_query: [b,1,c]
        latent_clip: [b*t,n,c]
        """
        BT, N, C = latent_clip.shape
        B = latent_query.shape[0]
        T = BT // B

        if self.cls_repair_neighbor:
            w = self.clip_feat_size_coarse
            feat = rearrange(latent_clip, '(b t) (h w) c -> b t h w c', b=B, t=T, w=w)
            neighbor_patches = torch.cat([
                rearrange(feat[:, :, 0:2, 2:4], '... hh ww c -> ... (hh ww) c'),  # [b,t,2,2,c] -> [b,t,4,c]
                rearrange(feat[:, :, 2:4, 0:4], '... hh ww c -> ... (hh ww) c'),  # [b,t,2,4,c] -> [b,t,8,c]
            ], dim=2)  # [b,t,12,c]
            neighbor_norm_mean = neighbor_patches.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)  # [b,t,1,1]
            neighbor_mean = neighbor_patches.mean(dim=2, keepdim=True)  # [b,t,1,c]
            neighbor_mean /= neighbor_mean.norm(dim=-1, keepdim=True)
            neighbor_mean *= neighbor_norm_mean
            neighbor_mean = rearrange(neighbor_mean, 'b t 1 c -> (b t) 1 c')
            latent_clip[:, [0, 1, w, w+1]] = neighbor_mean

        e = self.backbone.encoder
        if 'dino' in self.backbone_name:
            last_layer = e.layer[-1]
            Q_query = last_layer.attention.attention.query(latent_query)  # [b,1,c]
            K_clip = last_layer.attention.attention.key(latent_clip)  # [b*t,n,c]
        elif 'clip' in self.backbone_name:
            last_layer = e.layers[-1]
            Q_query = last_layer.self_attn.q_proj(latent_query)  # [b,1,c]
            K_clip = last_layer.self_attn.k_proj(latent_clip)

        Q_query = repeat(Q_query, 'b 1 c -> (b t) 1 c', t=T)
        attn = torch.bmm(Q_query, K_clip.transpose(1, 2)) / C ** 0.5  # [b*t,1,n]

        attn = F.sigmoid(attn - attn.mean(dim=-1, keepdim=True))  # [b*t,1,n]
        attn = attn.to(latent_query.dtype)

        attn = repeat(attn, '(b t) 1 n2 -> (b t H) n1 n2', b=B, H=self.nhead, n1=N, n2=N)

        return attn

    def compute_pca_score_map(self, guide_feat):
        b = guide_feat.shape[0]
        _feat = rearrange(guide_feat, 'b c h w -> b (h w) c')  # [b,h2*w2,c]
        _feat = _feat - _feat.mean(dim=1, keepdim=True)  # [b,h2*w2,c]
        score_maps = []  # [b,h2*w2,H]
        for bidx in range(b):
            U, S, V = torch.pca_lowrank(_feat[bidx], q=1+self.nhead)  # [h2*w2,1+H], [1+H], [c,1+H]
            score_map = torch.matmul(_feat[bidx], V[:, 1:])  # [h2*w2,c] @ [c,H] -> [h2*w2,H]
            score_maps.append(score_map)
        score_maps = torch.stack(score_maps)  # [b,h2*w2,H]
        score_maps = 1. - torch.exp(-1 * score_maps ** 2 / 1000)  # [b,h2*w2,H]
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

        elif self.backbone_name == 'clip-hf':
            b, _, h_origin, w_origin = x.shape
            x_forward_outs = self.backbone.forward(x, output_hidden_states=True)
            feat = x_forward_outs.last_hidden_state
            hidden_states = x_forward_outs.hidden_states[-1]
            cls_token = rearrange(feat[:, :1, :], 'b 1 c -> b c 1')
            feat = feat[:, 1:, :]
            h = int(h_origin / self.down_rate)
            w = int(w_origin / self.down_rate)
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

    def backbone_context(self):
        return InferenceContext(
            self.backbone_fp32_mm_precision,
            self.backbone_autocast,
            self.backbone_dtype,
            self.fix_backbone)

    def forward_conv_summary(self, clip_feat, output_dict, get_intermediate_features = False):
        for conv in self.conv_summary:
            clip_feat = clip_feat + conv(clip_feat)

        if get_intermediate_features:
            output_dict['feat']['clip']['conv'] = clip_feat.clone()

        return clip_feat

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
        output_dict = {'feat': {'clip': {}, 'query': {}, 'guide': {}}}

        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        with self.backbone_context():
            clip_feat_dict = self.extract_feature(segment)
            query_feat_dict = self.extract_feature(query)

        query_feat = query_feat_dict['feat']
        clip_feat = clip_feat_dict['feat']
        h, w = clip_feat_dict['h'], clip_feat_dict['w']

        # masks
        stx_tgt_mask = None   # [b*t*H,h*w,h*w], Q, K
        stx_mem_mask = None   # [b*t*H,h1*w1,h2*w2]

        # global guidance
        if self.enable_global_guidance:
            latent_query = query_feat_dict['hidden_states']
            latent_clip = clip_feat_dict['hidden_states']
            latent_query_cls = latent_query[:, :1]  # [b,1,c]
            latent_clip_non_cls = latent_clip[:, 1:]  # [b*t,n,c]

            cls_mask = self.get_cross_cls_attn_score(latent_query_cls, latent_clip_non_cls)
            stx_tgt_mask = cls_mask

        # local guidance
        if self.enable_local_guidance:
            score_maps = self.compute_pca_score_map(query_feat)  # [b,h2*w2,H]
            stx_mem_mask = repeat(score_maps, 'b (h2 w2) H -> (b t H) (h1 w1) (h2 w2)', t=t, h1=h, w1=w, h2=h, w2=w)

        # spatial correspondence
        query_feat_expanded = rearrange(query_feat, 'b c h w -> b (h w) c')  # [b,n,c]
        query_feat_expanded = query_feat_expanded.unsqueeze(1).expand(-1, t, -1, -1)  # [b,t,n,c]
        query_feat_expanded = rearrange(query_feat_expanded, 'b t (h w) c -> (b t) (h w) c', h=h)  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, '(b t) c h w -> (b t) (h w) c', b=b)  # [b*t,n,c]
        clip_feat = self.spatial_decoder.forward(
            tgt=clip_feat, tgt_mask=stx_tgt_mask,  # used in the SA block
            memory=query_feat_expanded, memory_mask=stx_mem_mask  # used in the CA block
        )  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, '(b t) (h w) c -> (b t) c h w', b=b, h=h)  # [b,t,c,h,w]

        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        # refine anchors
        anchors_yxhw = self.anchors_yxhw.to(device)                             # [N,4]
        anchors_yxyx = self.anchors_yxyx.to(device)                             # [N,4]
        anchors_yxhw = anchors_yxhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        anchors_yxyx = anchors_yxyx.reshape(1,1,-1,4)                           # [1,1,N,4]

        bbox_refine, prob = self.head.forward(clip_feat)                        # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        prob = self.logit_scale*rearrange(prob, '(b t) N c -> b t N c', b=b, t=t) # [b,t,N,1]
        prob = prob.squeeze(-1)                                                 # [b,t,N]
        bbox_refine += anchors_yxhw                                             # [b,t,N,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        pred_dict = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob,               # [b,t,N], logits
            'anchor': anchors_yxyx      # [1,1,N,4]
        }

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
                use_hnm=use_hnm,
            )

            loss_names = [k.replace('loss_', '') for k in loss_dict.keys() if 'loss_' in k]
            total_loss: torch.Tensor = torch.tensor(0., dtype=torch.float32, device=device, requires_grad=True)
            for loss_name in loss_names:
                ww, l = loss_dict[f'weight_{loss_name}'], loss_dict[f'loss_{loss_name}']
                if training:
                    assert l.requires_grad, f'{loss_name} should require grad'
                total_loss = total_loss + ww * l


            # Entropy Regularization
                # penalize entropy of a normed score map while limiting sigma's
                # entropy -> activate only specific parts
                # sigma -> limit the overall score magnitude
            loss_entropy = torch.tensor(0., dtype=query_feat.dtype, device=device)
            clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b)  # [b,t*h*w,c]
            query_feat = rearrange(query_feat.detach(), 'b c h w -> b (h w) c')  # [b,h*w,c]
            query_feat = query_feat - query_feat.mean(dim=1, keepdim=True)  # [b,h*w,c]
            for bidx in range(b):
                U, S, V = torch.pca_lowrank(clip_feat[bidx], q=self.rank_pca)  # [t*h*w,Q], [Q], [c,Q]
                score_map = torch.matmul(query_feat[bidx], V)  # [h*w,Q]

                score_map = 1. - torch.exp(-1 * score_map ** 2 / 10)  # [h*w,Q]
                score_map_normq = F.softmax(score_map / .2, dim=1)
                score_map_normq = score_map_normq.clamp(1e-6, 1-1e-6)
                score_map_normhw = F.softmax(score_map.mean(dim=0) / .1, dim=0)  # [Q]
                score_map_normhw = score_map_normhw.clamp(1e-6, 1-1e-6)

                patchwise_entropy = -(score_map_normq * score_map_normq.log()).sum(dim=1)  # [h*w,Q] -> [h*w]
                patchwise_entropy = patchwise_entropy.mean()
                mapwise_entropy = -(score_map_normhw * score_map_normhw.log()).sum()  # [Q] -> scalar
                loss_entropy = loss_entropy + patchwise_entropy - mapwise_entropy

            loss_entropy = loss_entropy / b
            total_loss = total_loss + self.weight_entropy * loss_entropy

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
        model = MATCH(config).cuda()
        print('Model with {} parameters'.format(sum(p.numel() for p in model.parameters())))
        print(config.model.cpt_path)
        checkpoint = torch.load(config.model.cpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        torch.set_grad_enabled(False)
        del checkpoint



    main()
