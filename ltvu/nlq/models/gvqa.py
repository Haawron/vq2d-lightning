import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Stack

from model.ours.nlq_head import NLQHead


def interp_env_feat(tensor, B, T_target, mode='nearest'):
    D, dtype = tensor.shape[-1], tensor.dtype
    return F.interpolate(tensor[None, None].float(), size=(B, T_target, D), mode=mode).squeeze([0, 1]).to(dtype=dtype)


class GroundVQA(nn.Module):
    def __init__(
        self, lm_path, input_dim, freeze_word = False, max_v_len = 256,
        enable_infuser = False,
        enable_infuser_activation = False,
        infuser_ca_mask = False,
        infuser_proper_v_masking = False,
        infuser_env_in_proj: None|int = None,  # None or input_dim
    ):
        super().__init__()
        self.enable_infuser = enable_infuser
        self.enable_infuser_activation = enable_infuser_activation
        self.infuser_ca_mask = infuser_ca_mask
        self.infuser_proper_v_masking = infuser_proper_v_masking
        self.infuser_env_in_proj = infuser_env_in_proj

        if not isinstance(input_dim, int):
            input_dim = input_dim.v_dim

        self.lm: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(lm_path, local_files_only=True)

        lm_dim = self.lm.get_input_embeddings().embedding_dim
        self.lm_proj = nn.Linear(input_dim, lm_dim)
        self.v_emb = nn.Parameter(torch.randn((1, 1, lm_dim)))
        if freeze_word:
            for name, param in self.lm.named_parameters():
                if 'shared' in name:
                    param.requires_grad = False

        self.nlq_head = NLQHead(in_dim=lm_dim, max_v_len=max_v_len)

        # ignore decoder
        self.lm.decoder = None
        self.lm.lm_head = None

        # infuser
        if enable_infuser:
            config = self.lm.config
            config.num_layers = 1
            config.is_decoder = True
            self.env_q_sbert_attn = T5Stack(config, self.lm.get_input_embeddings())
            self.gamma = nn.Parameter(torch.randn(1))
            self.vid_t_proj = nn.Linear(1200, 600)
            self.vid_sum_proj = nn.Linear(2 * lm_dim, lm_dim)
            self.vid_env_proj = nn.Linear(2 * lm_dim, lm_dim)
            if enable_infuser_activation:
                self.act = nn.GELU()
            else:
                self.act = nn.Identity()
            if infuser_env_in_proj is not None:
                assert isinstance(infuser_env_in_proj, int)
                self.env_in_proj = nn.Linear(infuser_env_in_proj, lm_dim)

    def forward(
        self,
        v_feat, v_mask, q_token, q_mask, gt_segments, gt_labels,
        env_feat=None, env_mask=None, q_feat=None,
        labels=None, **remains
    ):
        # encoder
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]

        # infuser
        if self.enable_infuser:
            encoder_out_v = self.forward_infuser(encoder_out_v, v_mask, env_feat, env_mask, q_feat)

        # localizer
        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            gt_segments=gt_segments,
            gt_labels=gt_labels
        )
        time_loss = nlq_results['final_loss'] * 1.0

        return time_loss, 0, time_loss

    def generate(self, v_feat, v_mask, q_token, q_mask, v_len, **remains):
        encoder_out, mask = self.forward_encoder(v_feat, v_mask, q_token, q_mask)
        encoder_out_v = encoder_out[:, -v_feat.shape[1]:]

        nlq_results = self.nlq_head(
            feat=encoder_out_v.permute(0, 2, 1),  # (B, D, T)
            mask=v_mask.unsqueeze(1),  # (B, 1, T)
            training=False,
            v_lens=v_len
        )

        answer_tokens = None

        return nlq_results, answer_tokens

    def forward_encoder(self, v_feat, v_mask, q_token, q_mask):
        B, L, D = v_feat.shape
        v_feat = self.lm_proj(v_feat)
        v_feat = v_feat + self.v_emb.expand((B, L, -1))
        q_feat = self.lm.encoder.embed_tokens(q_token)
        lm_input = torch.cat([q_feat, v_feat], dim=1)
        lm_mask = torch.cat([q_mask, v_mask], dim=1)
        out = self.lm.encoder(
            inputs_embeds=lm_input,
            attention_mask=lm_mask
        )
        return out.last_hidden_state, lm_mask

    def forward_infuser(self, encoder_out_v, v_mask, env_feat, env_mask, q_feat):
        B, T, D = encoder_out_v.shape

        if self.infuser_env_in_proj is not None:
            env_feat = self.env_in_proj(env_feat)  # [B, T_env, D]

        # env-query attention
        env_mask = env_mask if self.infuser_ca_mask else None
        q_feat = self.env_q_sbert_attn.forward(
            inputs_embeds=q_feat[:, None],  # [B, 1, D]
            attention_mask=None,
            encoder_hidden_states=env_feat,
            encoder_attention_mask=env_mask,
            use_cache=False,
            return_dict=True).last_hidden_state  # [B, 1, D]
        z_env = env_feat + F.tanh(self.gamma) * q_feat  # [B, T_env, D]

        # video summarization
        z_vid = res = encoder_out_v  # [B, T, D]
        z_vid = F.pad(z_vid, (0, 0, 0, 1200 - T), "constant", 0)  # [B, 1200, D]
        z_vid = rearrange(z_vid, 'b t d -> (b d) t')  # [B * D, 1200]
        if self.infuser_proper_v_masking:
            z_vid = z_vid * v_mask.unsqueeze(2)
        z_vid = self.vid_t_proj(z_vid)  # [B * D, proj_T]
        z_vid = rearrange(z_vid, '(b d) t -> b t d', b=B, d=D)  # [B, proj_T, D]
        z_vid = interp_env_feat(z_vid, B, T)  # [B, T, D]
        z_vid = z_vid * v_mask.unsqueeze(2)  # [B, T, D] <- [B, T, D] x [B, T, 1]
        z_vid = torch.cat([res, z_vid], dim=2)  # [B, T, 2D]
        z_vid = self.vid_sum_proj(z_vid)  # [B, T, D]
        z_vid = self.act(z_vid)
        z_vid = res + z_vid  # [B, T, D]

        # video-env fusion
        z_env = interp_env_feat(z_env, B, T)  # [B, T, D] <- [B, T_env, D]
        z_vid = torch.cat([z_vid, z_env], dim=2)  # [B, T, 2D]
        z_vid = self.vid_env_proj(z_vid)  # [B, T, D]
        z_vid = self.act(z_vid)
        z_vid = res + z_vid

        return z_vid
