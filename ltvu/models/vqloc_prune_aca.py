import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import torchvision
from transformers import Dinov2Model
import numpy as np

from ltvu.loss import get_losses_with_anchor
from ltvu.bbox_ops import bbox_xyhwToxyxy, generate_anchor_boxes_on_regions


base_sizes = torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    # 4 types of size
aspect_ratios = torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

# outputs = {}
# def hook_k(module, input, output):
#     outputs['desired_k'] = output

# def hook_q(module, input, output):
#     outputs['desired_q'] = output
    
def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio

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

def BasicBlock_Conv1D(in_dim, out_dim):
    module = nn.Sequential(
        nn.Conv1d(in_dim, out_dim, 1),
        nn.BatchNorm1d(out_dim),
        nn.LeakyReLU(inplace=True))
    return module


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
        reduce_ca_path=None,
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

        self.positive_threshold = positive_threshold
        self.logit_scale = logit_scale
        self.weight_bbox_center = weight_bbox_center
        self.weight_bbox_hw = weight_bbox_hw
        self.weight_bbox_giou = weight_bbox_giou
        self.weight_prob = weight_prob
        
        self.select_layer = -2 # args.mm_vision_select_layer
        self.reduction_ratio = 1/8
        self.left_tokens = (self.clip_feat_size_fine*self.clip_feat_size_coarse)*self.reduction_ratio

        self.anchors_xyhw = generate_anchor_boxes_on_regions(
            image_size=[self.clip_size_coarse, self.clip_size_coarse],
            num_regions=[self.num_anchor_regions, self.num_anchor_regions])
        self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)  # non-trainable, [R^2*N*M,4]

        if fix_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        if compile_backbone:
            self.backbone = torch.compile(self.backbone)

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
        # Conv2d에서 Conv1d로 변경하여 left_tokens에 적용
        # self.reduce = nn.Sequential(
        #     nn.Conv1d(self.backbone_dim, 256, kernel_size=1),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(256, 256, kernel_size=1),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(inplace=True),
        # )


        # clip-query correspondence
        self.CQ_corr_transformer = []
        for _ in range(1):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=transformer_dropout,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        if reduce_ca_path is not None:
            self.reduce_ca_load(reduce_ca_path)
            
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
        self.temporal_pe = torch.zeros(1, clip_num_frames, 256)
        self.temporal_pe = nn.parameter.Parameter(self.temporal_pe)

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

        # output head
        # self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.num_anchor_regions)
        self.head = Head(in_dim=256, left_tokens=self.left_tokens)
        
    def reduce_ca_load(self, reduce_ca_path):
        # Load the checkpoint
        state_dict = torch.load(reduce_ca_path, map_location='cpu', weights_only=True)

        # Load the weights for the 'reduce' and 'CQ_corr_transformer'
        reduce_weights = {k.replace('model.reduce.', ''): v for k, v in state_dict.items() if 'model.reduce' in k}
        transformer_weights = {k.replace('model.CQ_corr_transformer.', ''): v for k, v in state_dict.items() if 'model.CQ_corr_transformer' in k}

        # Load weights into the 'reduce' module
        reduce_state_dict = {k: v for k, v in reduce_weights.items() if k in self.reduce.state_dict()}
        self.reduce.load_state_dict(reduce_state_dict, strict=True)

        # Load weights into the 'CQ_corr_transformer' module
        for i, transformer_layer in enumerate(self.CQ_corr_transformer):
            layer_weights = {k.replace(f'{i}.', ''): v for k, v in transformer_weights.items() if f'{i}.' in k}
            transformer_layer.load_state_dict(layer_weights, strict=True)

        # Freeze the parameters in self.reduce and self.CQ_corr_transformer
        for param in self.reduce.parameters():
            param.requires_grad = False

        for param in self.CQ_corr_transformer.parameters():
            param.requires_grad = False

        print("Reduce and CQ_corr_transformer Weights loaded and layers frozen.")

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def extract_feature(self, x, return_h_w=False, token_prune=False, return_cls=False) -> torch.Tensor | tuple[torch.Tensor, int, int]:
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
        elif self.backbone_name in ['dinov2', 'dinov2-hf']:
            b, _, h_origin, w_origin = x.shape
            if 'hf' in self.backbone_name:
                if token_prune:
                    out, idx = self.token_prune_merge_advanced_plus_mat(x, if_adaptive=True, reduction_ratio=1/8) 
                else:
                    out = self.backbone.forward(x).last_hidden_state
                    cls_out = rearrange(out[:,:1,:],'b n c -> b c n')
                    out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
                h = int(h_origin / self.down_rate)
                w = int(w_origin / self.down_rate)
            else:
                out = self.backbone.get_intermediate_layers(x, n=1)[0]
                h = int(h_origin / self.backbone.patch_embed.patch_size[0])
                w = int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2) if not token_prune else out.reshape(b, -1, dim).permute(0,2,1) # [b,c,h,w] [b,c,left_tokens]
        else:
            raise NotImplementedError

        if torch.isnan(out).any():
            raise ValueError('nan in feature')
        out = out.float()
        if token_prune:
            if return_h_w:
                return out, idx, h, w
            return out, idx
        if return_h_w:
            if return_cls:
                return out, cls_out, h, w
            return out, h, w
        if return_cls:
            return out, cls_out
        return out

    def get_mask(self, src, t):
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
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer] # penultimate layer output
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features


    def token_prune_merge_advanced_plus(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
        version 24/03/2024 using the spacially smapled tokens to supplement the pruned tokens
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.backbone.encoder.layer[-1].attention.attention.query.register_forward_hook(hook_k)
        hook_handle_q = self.backbone.encoder.layer[-1].attention.attention.query.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.backbone.forward(images, output_hidden_states=True)
        # image_forward_outs = self.backbone(images).last_hidden_state
        image_features =image_forward_outs.hidden_states[self.select_layer][:, 1:] # cls none, only patch
        # image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
            self.reduction_ratio = reduction_ratio
        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        
        self.device = images.device
        # # # print("idx: ", idx)
        # if if_adaptive:
        #     step_length = int(1/reduction_ratio)
        #     arithmetic_sequence = torch.arange(0, 575, int(step_length/3)).to(device=self.device)
        #     original_tensor_1d = idx.flatten().to(device=self.device)
        #     filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
        #     concatenated_tensor = torch.cat((idx, filtered_sequence.unsqueeze(0)), dim=1)
        #     idx = concatenated_tensor
        #     # # print("idx_new: ", idx)
        # else:
        #     # # this is for training
        #     step_length = int(1/reduction_ratio)
        #     new_idx = torch.zeros((idx.size(0), idx.size(1)*2), dtype=torch.long).to(device=self.device)
        #     for i in range(idx.size(0)):
        #         arithmetic_sequence = torch.arange(int(step_length/2), 575, int(step_length)).to(device=self.device)
        #         original_tensor_1d = idx[i].flatten().to(device=self.device)
        #         filtered_sequence = arithmetic_sequence
        #         # filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
        #         concatenated_tensor = torch.cat((original_tensor_1d, filtered_sequence), dim=0)
        #         new_idx[i] = concatenated_tensor
        #     idx = new_idx

        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = x_others[b, i, :]  + weighted_avg 
                updated_x_others[b, i, :] = updated_center 
            
        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        # updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
        image_features = updated_x_others
        return image_features, idx

    def token_prune_merge_advanced_plus_mat(self, image_features, clip_feat, query_feat, if_adaptive=True, reduction_ratio=1/8):
        '''
        Optimized version - batch processing and minimized loops for efficiency.
        '''
        
        for layer in self.CQ_corr_transformer:
            # Self-attention 수행
            attn_output_self, attn_weights_self = layer.self_attn(clip_feat, clip_feat, clip_feat)
            
            # MultiheadAttention의 속성 가져오기
            cross_attn = layer.multihead_attn
            embed_dim = cross_attn.embed_dim  # 일반적으로 256
            num_heads = cross_attn.num_heads  # 일반적으로 4
            head_dim = embed_dim // num_heads
            scaling = float(head_dim) ** -0.5

            # in_proj_weight와 in_proj_bias로부터 q, k 추출
            q_proj_weight = cross_attn.in_proj_weight[:embed_dim, :]
            k_proj_weight = cross_attn.in_proj_weight[embed_dim:2*embed_dim, :]
            q_proj_bias = cross_attn.in_proj_bias[:embed_dim]
            k_proj_bias = cross_attn.in_proj_bias[embed_dim:2*embed_dim]

            # 선형 변환 수행
            desired_layer_q = F.linear(attn_output_self, q_proj_weight, q_proj_bias) * scaling
            desired_layer_k = F.linear(query_feat, k_proj_weight, k_proj_bias)

            # Attention Score 계산
            attn_scores = torch.matmul(desired_layer_q, desired_layer_k.transpose(-2, -1))
            attn_probs = F.softmax(attn_scores, dim=-1)


        B, N, C = image_features.shape
        
        cls_attn = attn_probs.mean(dim=1, keepdim=False)        
        # cls_attn = attn[:, 0, 1:]  # Attention weights for non-CLS tokens
        
        # Adaptive reduction ratio based on attention
        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)
            self.reduction_ratio = reduction_ratio

        _, idx = torch.topk(cls_attn, int(N * reduction_ratio), dim=1, largest=True)  # Top-k tokens

        # Handle batched token gathering
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        # print(f'index {index.shape}, attn {attn.shape}, image_features {image_features.shape}, desired_layer {desired_layer_k.shape}, {desired_layer_q.shape}')
        
        # Batch gather features and keys
        Key_wo_cls = desired_layer_k  # Remove CLS token for keys
        x_others = torch.gather(image_features, dim=1, index=index)  # Top-k features
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  # Top-k attention
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # Top-k keys

        # Handle non-top-k tokens
        compl = complement_idx(idx, N)  # Indices of non-top-k tokens
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # Non-top-k features
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # Non-top-k keys
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # Non-top-k attention

        # Normalize keys for cosine similarity calculation
        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # Batch cosine similarity and top-k clustering
        cos_sim_matrix = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2))  # [B, left_tokens, N-1-left_tokens]
        _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)  # Top-k clusters per token

        # # Gather cluster tokens and calculate weighted averages in batches
        # cluster_tokens = torch.gather(non_topk, dim=1, index=cluster_indices.unsqueeze(-1).expand(-1, -1, C))
        # weights = torch.gather(non_topk_attn, dim=1, index=cluster_indices).unsqueeze(-1)

        # weighted_avg = torch.sum(cluster_tokens * weights, dim=2)  # Batch weighted average of clusters

        # # Update centers in batch
        # updated_x_others = x_others + weighted_avg
        
        B, left_tokens, cluster_size = cluster_indices.shape  # B=96, left_tokens=160, cluster_size=32
        _, non_topk_size, C = non_topk.shape  # non_topk_size=864, C=768

        # Expanding `cluster_indices` for gathering tokens in the feature dimension
        expanded_cluster_indices = cluster_indices.unsqueeze(-1).expand(B, left_tokens, cluster_size, C)  # [96, 160, 32, 768]
        cluster_tokens = torch.gather(non_topk.unsqueeze(1).expand(-1, left_tokens, -1, -1), dim=2, index=expanded_cluster_indices)  # [96, 160, 32, 768]

        # Now gather attention values from `non_topk_attn`
        expanded_attn_indices = cluster_indices  # Shape [96, 160, 32]
        cluster_attn = torch.gather(non_topk_attn.unsqueeze(1).expand(-1, left_tokens, -1), dim=2, index=expanded_attn_indices)  # [96, 160, 32]

        # Compute weighted average of clusters
        weights = cluster_attn.unsqueeze(-1)  # Shape [96, 160, 32, 1]
        weighted_avg = torch.sum(cluster_tokens * weights, dim=2)  # Shape [96, 160, 768] - sum over clusters

        # Update top-k tokens
        updated_x_others = x_others + weighted_avg  # Shape [96, 160, 768]

        # Compute extra token for residual tokens
        # extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]

        # Return updated features and indices
        return updated_x_others, idx


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
        segment = rearrange(segment, 'b t c h w -> (b t) c h w')
        device = segment.device

        # get backbone features
        with torch.no_grad():
            prec_prev = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision(self.backbone_fp32_mm_precision)
            with torch.autocast(device_type="cuda", dtype=self.backbone_dtype, enabled=self.backbone_autocast):
                query_feat, query_cls, h, w = self.extract_feature(query, return_h_w=True, return_cls=True)    # [b,c,h,w]
                clip_feat = self.extract_feature(segment, token_prune=False)   # [b*t,c,h,w]
            torch.set_float32_matmul_precision(prec_prev)

        # reduce channel size
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_cls = self.reduce(query_cls.unsqueeze(-1))
        # query_feat = rearrange(query_feat, 'b c h w -> b c (h w)')
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)

        if use_hnm and compute_loss:
            clip_feat, query_feat = self.replicate_for_hnm(query_feat, clip_feat)   # b -> b^2
            b **= 2

        # find spatial correspondence between query-frame
        # query_feat = repeat(query_feat, 'b c h w -> (b t) (h w) c', t=t)  # [b*t,n,c] # DEBUG: same as below
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')# [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')            # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            after_clip_feat = layer(clip_feat, query_feat)                        # [b*t,n,c]
        # after_clip_feat = rearrange(after_clip_feat, 'b (h w) c -> b c h w', h=h, w=w)  # [b*t,c,h,w]

        # [b*t,c,left_tokens], [b*t,left_tokens]
        clip_feat, idx = self.token_prune_merge_advanced_plus_mat(after_clip_feat, clip_feat, query_feat, if_adaptive=True, reduction_ratio=1/8) 
        
        # down-size features and find spatial-temporal correspondence
        # for head in self.down_heads:
        #     clip_feat = head(clip_feat)
        #     if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
        clip_feat = rearrange(clip_feat, '(b t) left_tokens c -> b left_tokens t c', b=b) + self.temporal_pe
        clip_feat = rearrange(clip_feat, 'b left_tokens t c -> b (t left_tokens) c')
        mask = self.get_mask(clip_feat, t)
        for layer in self.feat_corr_transformer:
            clip_feat = layer(clip_feat, src_mask=mask)
        clip_feat = rearrange(clip_feat, 'b (t left_tokens) c -> (b t) c left_tokens', b=b, t=t) #, h=self.resolution_transformer, w=self.resolution_transformer)
        # break  # ? why break here

        # refine anchors
        anchors_xyhw = self.anchors_xyhw.to(device)                             # [N,4]
        anchors_xyxy = self.anchors_xyxy.to(device)                             # [N,4]
        anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           # [1,1,N,4]


        expanded_idx = rearrange(idx, '(b t) left_tokens -> b t left_tokens', b=b)   # [B, 1, left_tokens, 1]
        expanded_idx = expanded_idx.unsqueeze(-1) * 12                               # [B, t, left_tokens, 1]
        index_range = torch.arange(12, device=expanded_idx.device).view(1, 1, 1, -1) # [1, 1, 1, 12]
        full_idx = expanded_idx + index_range                                        # [B, t, left_tokens, 12]
        full_idx = rearrange(full_idx,'b t n a -> b t (n a)').unsqueeze(-1).expand(-1,-1,-1,4) # [B, t, left_tokens*12, 12]
        selected_anchors_xyhw = torch.gather(anchors_xyhw.expand(b, t, -1, 4), dim=2, index=full_idx)
        selected_anchors_xyxy = torch.gather(anchors_xyxy.expand(b, t, -1, 4), dim=2, index=full_idx)

        bbox_refine, prob = self.head.forward(clip_feat)                        # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        prob = self.logit_scale*rearrange(prob, '(b t) N c -> b t N c', b=b, t=t) # [b,t,N,1]
        prob = prob.squeeze(-1)                                                 # [b,t,N]
        bbox_refine += selected_anchors_xyhw                                             # [b,t,N,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        pred_dict = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob,               # [b,t,N], logits
            'anchor': selected_anchors_xyxy      # [1,1,N,4]
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

            # for logging - losses
            loss_dict = detach_dict(loss_dict)
            log_dict = {
                'loss': total_loss.detach(),
                'loss_bbox_center': loss_dict['loss_bbox_center'].mean(),
                'loss_bbox_hw': loss_dict['loss_bbox_hw'].mean(),
                'loss_bbox_giou': loss_dict['loss_bbox_giou'].mean(),
                'loss_prob': loss_dict['loss_prob'].mean(),
            }

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
                'reduction_ratio': self.reduction_ratio,  # total patch reduction_ratio
                'patch_num': int(N*self.reduction_ratio),  # total patch reduction_ratio
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
    def __init__(self, in_dim=256, left_tokens=64, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.left_tokens = left_tokens
        self.num_layers = 3

        self.in_conv = BasicBlock_Conv1D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv1D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv1D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv1d(in_dim, n * m * 4, kernel_size=1)
        self.classification_head = nn.Conv1d(in_dim, n * m * 1, kernel_size=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv1d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B,c,h=8,w=8]
        '''

        B, c, left_tokens = x.shape

        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,left_tokens]
        # dpout pos 1, seems better
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,left_tokens]
        feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,left_tokens]

        out_reg = self.regression_head(feat_reg)
        out_cls = self.classification_head(feat_cls)

        out_reg = rearrange(out_reg, 'B (n m c) left_tokens -> B (left_tokens n m) c', n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) left_tokens -> B (left_tokens n m) c', n=self.n, m=self.m, c=1)

        return out_reg, out_cls
    
# class Head(nn.Module):
#     def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
#         super(Head, self).__init__()

#         self.in_dim = in_dim
#         self.n = n
#         self.m = m
#         self.num_up_layers = int(math.log2(out_res // in_res))
#         self.num_layers = 3

#         if self.num_up_layers > 0:
#             self.up_convs = []
#             for _ in range(self.num_up_layers):
#                 self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
#             self.up_convs = nn.Sequential(*self.up_convs)

#         self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

#         self.regression_conv = []
#         for i in range(self.num_layers):
#             self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
#         self.regression_conv = nn.Sequential(*self.regression_conv)

#         self.classification_conv = []
#         for i in range(self.num_layers):
#             self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
#         self.classification_conv = nn.Sequential(*self.classification_conv)

#         self.droupout_feat = torch.nn.Dropout(p=0.2)
#         self.droupout_cls = torch.nn.Dropout(p=0.2)

#         self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
#         self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

#         self.regression_head.apply(self.init_weights_conv)
#         self.classification_head.apply(self.init_weights_conv)

#     def init_weights_conv(self, m):
#         if type(m) == nn.Conv2d:
#             nn.init.normal_(m.weight, mean=0.0, std=1e-6)
#             nn.init.normal_(m.bias, mean=0.0, std=1e-6)

#     def forward(self, x):
#         '''
#         x in shape [B,c,h=8,w=8]
#         '''
#         if self.num_up_layers > 0:
#             x = self.up_convs(x)     # [B,c,h=16,w=16]

#         B, c, h, w = x.shape

#         feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w]
#         # dpout pos 1, seems better
#         feat_reg = self.droupout_feat(feat_reg)
#         feat_cls = self.droupout_cls(feat_cls)

#         feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,h,w]
#         feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,h,w]

#         out_reg = self.regression_head(feat_reg)
#         out_cls = self.classification_head(feat_cls)

#         out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
#         out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

#         return out_reg, out_cls


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
