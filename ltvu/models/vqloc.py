import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import torchvision
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
    # TODO: 이거 허깅페이스로 바까버려
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
    elif backbone_name == 'dinov2-hf':
        backbone = Dinov2Model.from_pretrained('facebook/dinov2-base')
        down_rate = 14
        backbone_dim = 768
    return backbone, down_rate, backbone_dim


def BasicBlock_Conv2D(in_dim, out_dim):
    module = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True))
    return module


class ClipMatcher(nn.Module):
    def __init__(self,
        backbone_name = 'dinov2',
        backbone_type = 'vitb14',
        fix_backbone = True,
        window_transformer = 5,  # sec
        resolution_transformer = 8,
        num_anchor_regions = 16,
        num_layers_st_transformer = 3,

        query_size = 448,
        clip_size_fine = 448,
        clip_size_coarse = 448,
        clip_num_frames = 30,
    ) -> None:
        super().__init__()

        self.backbone, self.down_rate, self.backbone_dim = build_backbone(backbone_name, backbone_type)
        self.backbone_name = backbone_name

        self.query_size = query_size
        self.clip_size_fine = clip_size_fine
        self.clip_size_coarse = clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.window_transformer = window_transformer
        self.resolution_transformer = resolution_transformer
        self.num_anchor_regions = num_anchor_regions

        self.anchors_xyhw = generate_anchor_boxes_on_regions(
            image_size=[self.clip_size_coarse, self.clip_size_coarse],
            num_regions=[self.num_anchor_regions, self.num_anchor_regions])
        self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)

        if fix_backbone:  # TODO: compile
            for param in self.backbone.parameters():
                param.requires_grad = False

        # query down heads
        self.query_down_heads = []
        for _ in range(int(math.log2(self.query_feat_size))):
            self.query_down_heads.append(
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.query_down_heads = nn.ModuleList(self.query_down_heads)

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
        for _ in range(1):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        # feature downsample layers
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(self.num_head_layers-1):
            self.in_channel = 256 if i != 0 else self.backbone_dim
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
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None

        # output head
        self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.num_anchor_regions)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out

    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
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


    def forward(
        self,
        segment,
        query,
        before_query_mask: torch.Tensor = None,
        gt_bboxes = None,
        gt_probs = None,
        compute_loss = False,
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
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')

        # get backbone features
        query_feat = self.extract_feature(query)        # [b c h w]
        clip_feat = self.extract_feature(segment)          # (b t) c h w
        h, w = clip_feat.shape[-2:]

        # reduce channel size
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if use_hnm and compute_loss:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   # b -> b^2
            b **= 2

        # find spatial correspondence between query-frame
        query_feat = repeat(query_feat, 'b c h w -> (b t) (h w) c', t=t)  # [b*t,n,c] # DEBUG: same as below
        # query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')# [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')            # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)                        # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)  # [b*t,c,h,w]

        # down-size features and find spatial-temporal correspondence
        for head in self.down_heads:
            clip_feat = head(clip_feat)
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break  # ? why break here

        # refine anchors
        anchors_xyhw = self.anchors_xyhw.to(clip_feat.device)                   # [N,4]
        anchors_xyxy = self.anchors_xyxy.to(clip_feat.device)                   # [N,4]
        anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           # [1,1,N,4]

        bbox_refine, prob = self.head(clip_feat)                                # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                # [b,t,N,1]
        bbox_refine += anchors_xyhw                                             # [b,t,N,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        pred_dict = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob.squeeze(-1),   # [b,t,N]
            'anchor': anchors_xyxy      # [1,1,N,4]
        }
        output_dict = {'pred_dict': pred_dict}

        if compute_loss:
            gts = {
                'before_query': before_query_mask,  # [b,t]
                'clip_with_bbox': gt_probs,         # [b,t]
                'clip_bbox': gt_bboxes,             # [b,t,4]
                # 'hw': None,                     # [b,t,2]
                # 'center': None,                 # [b,t,2]
            }
            # acutal loss calculation
            loss_dict, preds_top, gts = get_losses_with_anchor(pred_dict, gts)
            total_loss = sum(loss_dict[k.replace('loss_', 'weight_')] * v for k, v in loss_dict.items() if 'loss_' in k)
            output_dict.update({'loss': total_loss})

            # for logging
            loss_dict = detach_dict(loss_dict)
            preds_top = detach_dict(preds_top)
            log_dict = {
                'loss': total_loss.detach(),
                'loss_bbox_center': loss_dict['loss_bbox_center'].mean(),
                'loss_bbox_hw': loss_dict['loss_bbox_hw'].mean(),
                'loss_bbox_giou': loss_dict['loss_bbox_giou'].mean(),
                'loss_prob': loss_dict['loss_prob'].mean(),
                'iou': loss_dict['iou'].mean(),
                'giou': loss_dict['giou'].mean(),
            }
            output_dict.update({'log_dict': log_dict})
            info_dict = {
                'loss_dict': loss_dict,     # losses starting with 'loss_', weights starting with 'weight_', iou, giou
                'preds_top': preds_top,     # bbox: [b,t,4], prob: [b,t]
                'gts': gts,                 # gts with hw, center computed
            }
            output_dict.update({'info_dict': info_dict})

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

        # dpout pos 2

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
