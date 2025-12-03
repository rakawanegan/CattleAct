import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stgcn import Model


class ViTBackbone(nn.Module):
    """
    torchvision ViT-B/16 wrapper
    output:
      - cls:        [B, D]
      - tokens:     [B, Gh*Gw, D]
      - grid_size:  (Gh, Gw)
      - patch_size: int
      - feat_hw:    (H_in, W_in)  # ViT input image size
      - embed_dim:  D
    """

    def __init__(self, arch="vit_b_16", weights=models.ViT_B_16_Weights.IMAGENET1K_V1):
        super().__init__()
        vit = getattr(models, arch)(weights=weights)
        self.vit = vit
        # 解析超参
        self.patch_size = vit.conv_proj.kernel_size[0]
        self.feat_hw = (vit.image_size, vit.image_size)
        self.grid_size = (
            vit.image_size // self.patch_size,
            vit.image_size // self.patch_size,
        )
        self.embed_dim = vit.hidden_dim

    # @torch.no_grad()
    def forward_features(self, x: torch.Tensor):
        vit = self.vit
        B = x.size(0)
        x = vit._process_input(x)  # -> [B, Gh*Gw, D]
        n = x.shape[0]
        batch_class_token = vit.class_token.expand(n, -1, -1)  # [B,1,D]
        x = torch.cat([batch_class_token, x], dim=1)  # [B, 1+N, D]
        x = x + vit.encoder.pos_embedding
        x = vit.encoder.dropout(x)
        x = vit.encoder.layers(x)  # Transformer encoder
        x = vit.encoder.ln(x)  # [B, 1+N, D]
        cls = x[:, 0]  # [B, D]
        tokens = x[:, 1:]  # [B, N, D]（w/o CLS）
        return cls, tokens

    def forward(self, x):
        cls, tokens = self.forward_features(x)
        return {
            "cls": cls,
            "tokens": tokens,
            "grid_size": self.grid_size,
            "patch_size": self.patch_size,
            "feat_hw": self.feat_hw,
            "embed_dim": self.embed_dim,
        }


class JointPatchBilinearAlign(nn.Module):
    """
    put joint pixel coordinates into patch tokens
    Input:
      - tokens:   [B, Np, D]
      - joints_xy:[B, V, 2]  absolute pixel coordinates of joints
      - grid_size:(Gh,Gw), patch_size=P, feat_hw=(H_in,W_in)
    Output:
      - joint_img_tok: [B, V, D]
    """

    def __init__(self):
        super().__init__()

    def forward(self, tokens, joints_xy, grid_size, patch_size, feat_hw):
        B, Np, D = tokens.shape
        Gh, Gw = grid_size
        P = patch_size
        H_in, W_in = feat_hw

        # mesh grid
        y_hat = (joints_xy[..., 1] / P).clamp(0, Gh - 1 - 1e-6)  # [B,V]
        x_hat = (joints_xy[..., 0] / P).clamp(0, Gw - 1 - 1e-6)

        i0 = torch.floor(y_hat)
        j0 = torch.floor(x_hat)
        di = y_hat - i0
        dj = x_hat - j0
        i1 = (i0 + 1).clamp(0, Gh - 1)
        j1 = (j0 + 1).clamp(0, Gw - 1)

        i0 = i0.long()
        j0 = j0.long()
        i1 = i1.long()
        j1 = j1.long()

        idx00 = i0 * Gw + j0  # [B,V]
        idx01 = i0 * Gw + j1
        idx10 = i1 * Gw + j0
        idx11 = i1 * Gw + j1

        def gather_index(idx):
            # tokens: [B, Np, D], idx: [B,V] -> [B,V,D]
            return tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

        t00 = gather_index(idx00)
        t01 = gather_index(idx01)
        t10 = gather_index(idx10)
        t11 = gather_index(idx11)

        w00 = (1 - di) * (1 - dj)
        w01 = (1 - di) * dj
        w10 = di * (1 - dj)
        w11 = di * dj

        joint_img_tok = (
            w00.unsqueeze(-1) * t00
            + w01.unsqueeze(-1) * t01
            + w10.unsqueeze(-1) * t10
            + w11.unsqueeze(-1) * t11
        )  # [B,V,D]
        return joint_img_tok


class STGCNBackbone(nn.Module):

    def __init__(self, stgcn_model_ctor, model_kwargs):
        super().__init__()
        # No classfier, only GCN backbone
        mk = dict(model_kwargs)
        mk["without_fc"] = True
        self.gcn = stgcn_model_ctor(
            **mk
        )  # e.g., Model(data_shape=..., num_classes=..., without_fc=True)
        self.out_dim = self.gcn.out_channels

    def forward(self, skel):  # skel: [N, C, T, V, M]
        feat = self.gcn(skel)  # [N, Cgcn]
        return feat


class ViT_STGCN_Fusion(nn.Module):
    """
    - ViT global CLS + ST-GCN global features -> global fusion
    - (optional) local joint : joint-patch token concat with pose -> small MLP -> pose pooling
    """

    def __init__(self, num_classes, gcn, use_local=True, drop=0.1):
        super().__init__()
        self.use_local = use_local

        # setting
        self.vit = ViTBackbone()
        self.align = JointPatchBilinearAlign()

        self.gcn = gcn
        self.proj_pose_global = nn.Linear(self.gcn.output_dim, 512)

        d_img = self.vit.embed_dim

        self.proj_img_global = nn.Linear(d_img, 512)
        self.proj_pose_global = nn.Linear(self.gcn.output_dim, 512)

        # local pose MLP
        if self.use_local:
            in_local = d_img
            self.local_mlp = nn.Sequential(
                nn.LayerNorm(in_local),
                nn.Linear(in_local, 256),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(256, 256),
            )

        fused_dim = 512 * 2 + (256 if self.use_local else 0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, skel, joints_xy=None, vis_mask=None):
        """
        img:            [B, 3, H, W]
        skel:           [B, C, T, V, M]
        joints_xy:      [B, V, 2]
        vis_mask:       [B, V]    # visibility mask for local pose (only use important joints?)
        """
        assert self.gcn is not None, "Call set_gcn() first."

        # 1) ViT  & patch tokens
        vit_out = self.vit(img)
        cls = vit_out["cls"]  # [B, Dimg]
        tokens = vit_out["tokens"]  # [B, Np, Dimg]

        # 2) ST-GCN backbone
        gcn_feat = self.gcn.extract_feature(skel)  # [B, Dgcn]

        # 3) global fusion
        g_img = self.proj_img_global(cls)  # [B,512]
        g_pose = self.proj_pose_global(gcn_feat)  # [B,512]
        g_fused = torch.cat(
            [g_img, g_pose], dim=1
        )  # [B,1024]  before classifier -> LayerNorm+Linear->256

        # 4) local fusion（optional）
        if self.use_local and (joints_xy is not None):
            joint_local = self.align(
                tokens,
                joints_xy,
                vit_out["grid_size"],
                vit_out["patch_size"],
                vit_out["feat_hw"],
            )  # [B,V,Dimg]

            z = self.local_mlp(joint_local)  # [B,V,256]

            if vis_mask is not None:
                w = vis_mask.float().unsqueeze(-1)  # [B,V,1]
                z = z * w
                denom = w.sum(dim=1).clamp_min(1.0)  # [B,1]
                z = z.sum(dim=1) / denom  # [B,256]
            else:
                z = z.mean(dim=1)  # [B,256]
            fused_vec = torch.cat(
                [g_fused, z], dim=1
            )  # [B, 512+512? -> 512 + 256 after proj]
        else:
            fused_vec = g_fused[:, :512] + g_fused[:, 512:]  # [B, 512]
            fused_vec = torch.cat(
                [fused_vec, torch.zeros_like(fused_vec[:, :256])], dim=1
            )  # pad to [... , 512+256]

        logits = self.classifier(fused_vec)  # [B,num_classes]
        return logits


if __name__ == "__main__":
    # Example usage
    gcn_model = Model(
        in_channels=3,
        num_class=3,
        graph_args={"layout": "cattle", "strategy": "spatial"},
        edge_importance_weighting=False,
    )
    model = ViT_STGCN_Fusion(num_classes=4, gcn=gcn_model, use_local=True)
    img = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    skel = torch.randn(2, 3, 1, 17, 1)  # Batch of 2 skeletons (C,T,V,M)
    joints_xy = torch.randn(2, 17, 2)  # Batch of 2 sets of joint coordinates
    output = model(img, skel, joints_xy)
    print(output.shape)
