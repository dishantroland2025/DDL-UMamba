# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DDL_CC_3D(nn.Module):
#     def __init__(self, channels, dv=3, reduction=4):
#         """
#         Expanded-State Deep Delta Learning Block with Channel Convolution (DDL-CC)
        
#         Args:
#             channels (int): Input channel dimension (C)
#             dv (int): State expansion factor (default: 3 for RV, MYO, LV slots)
#             reduction (int): Reduction factor for internal context generation
#         """
#         super().__init__()
#         self.channels = channels
#         self.dv = dv
#         self.expanded_dim = channels * dv
#         hidden_dim = max(self.expanded_dim // reduction, 16)

#         # -----------------------------------------------------------
#         # Context & Geometric Heads
#         # -----------------------------------------------------------
#         self.context_gen = nn.Sequential(
#             nn.Conv3d(self.expanded_dim, hidden_dim, kernel_size=1),
#             nn.InstanceNorm3d(hidden_dim),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
#         )

#         self.k_head = nn.Conv3d(hidden_dim, channels, kernel_size=1)
#         self.beta_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
#         self.v_head = nn.Conv3d(hidden_dim, dv, kernel_size=1)

#         # -----------------------------------------------------------
#         # TRUE PAPER-FAITHFUL CC COMPRESSION
#         # -----------------------------------------------------------
#         # Conv1d along the dv axis: in_channels=C, out_channels=C, groups=C, kernel_size=dv
#         # bias=False explicitly enforced to maintain mathematical purity of the identity map.
#         self.compress_cc = nn.Conv1d(
#             in_channels=channels, 
#             out_channels=channels, 
#             kernel_size=dv, 
#             groups=channels,
#             bias=False
#         )

#         self._init_weights()

#     def _init_weights(self):
#         # Beta initialization guarantees identity map at start
#         nn.init.constant_(self.beta_head.bias, -5.0)
#         nn.init.xavier_normal_(self.beta_head.weight, gain=0.01)

#         # Orthogonal init for the reflection direction
#         nn.init.orthogonal_(self.k_head.weight)
#         if self.k_head.bias is not None:
#             nn.init.constant_(self.k_head.bias, 0)
            
#         # Initialize CC Conv1d to average the dv slots perfectly
#         nn.init.constant_(self.compress_cc.weight, 1.0 / self.dv)

#     def forward(self, x):
#         B, C, D, H, W = x.shape
        
#         # 1. Expansion
#         x_flat = x.repeat_interleave(self.dv, dim=1)

#         # 2. Context Generation
#         ctx = self.context_gen(x_flat)

#         # 3. Geometric Parameters
#         k_raw = self.k_head(ctx) 
#         k = F.normalize(k_raw, p=2, dim=1, eps=1e-6)
        
#         beta = 2.0 * torch.sigmoid(self.beta_head(ctx)) 
#         v = self.v_head(ctx) 
        
#         # 4. Exact matrix-valued Delta Update (Eq. 2.5)
#         X_mat = x_flat.view(B, C, self.dv, D, H, W)
        
#         k_mat = k.unsqueeze(2)       
#         v_mat = v.unsqueeze(1)       
#         beta_mat = beta.unsqueeze(2) 
        
#         k_T_X = torch.sum(k_mat * X_mat, dim=1, keepdim=True)
#         update = beta_mat * k_mat * (v_mat - k_T_X)
#         X_updated = X_mat + update 
        
#         # -----------------------------------------------------------
#         # 5. PAPER-FAITHFUL CC COMPRESSION (Conv1d on dv axis)
#         # -----------------------------------------------------------
#         # Permute to: (B, D, H, W, C, dv)
#         X_transposed = X_updated.permute(0, 3, 4, 5, 1, 2)
        
#         # Reshape to: (B*D*H*W, C, dv)
#         X_reshaped = X_transposed.reshape(B * D * H * W, C, self.dv)
        
#         # Apply Channel Convolution along the dv axis
#         out_cc = self.compress_cc(X_reshaped)
        
#         # Unfold back to 3D spatial format: (B, C, D, H, W)
#         out = out_cc.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

#         return out, k, beta, update






import torch
import torch.nn as nn
import torch.nn.functional as F

class DDL_CC_3D(nn.Module):
    def __init__(self, channels, dv=4, reduction=4):
        """
        Expanded-State Deep Delta Learning Block with Channel Convolution (DDL-CC)
        Constrained Version: L2 Weight Norm (prevents parameter bloat) + Sigmoid (preserves Theorem 3.1 bounds)
        """
        super().__init__()
        self.channels = channels
        self.dv = dv
        self.expanded_dim = channels * dv
        hidden_dim = max(self.expanded_dim // reduction, 16)

        # -----------------------------------------------------------
        # Context & Geometric Heads
        # -----------------------------------------------------------
        self.context_gen = nn.Sequential(
            nn.Conv3d(self.expanded_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm3d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.k_head = nn.Conv3d(hidden_dim, channels, kernel_size=1)
        self.beta_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        self.v_head = nn.Conv3d(hidden_dim, dv, kernel_size=1)

        # -----------------------------------------------------------
        # CC COMPRESSION
        # -----------------------------------------------------------
        self.compress_cc = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=dv, 
            groups=channels,
            bias=False
        )

        self._init_weights()

    def _init_weights(self):
        # Beta initialization at -5.0 guarantees 2 * Sigmoid starts at ~0.013 (Identity map warmup)
        nn.init.constant_(self.beta_head.bias, -5.0)
        nn.init.xavier_normal_(self.beta_head.weight, gain=0.01)

        # Orthogonal init for the reflection direction
        nn.init.orthogonal_(self.k_head.weight)
        if self.k_head.bias is not None:
            nn.init.constant_(self.k_head.bias, 0)
            
        # Initialize CC Conv1d to average the dv slots perfectly
        nn.init.constant_(self.compress_cc.weight, 1.0 / self.dv)

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # 1. Expansion
        x_flat = x.repeat_interleave(self.dv, dim=1)

        # 2. Context Generation
        ctx = self.context_gen(x_flat)

        # 3. Geometric Parameters
        k_raw = self.k_head(ctx) 
        k = F.normalize(k_raw, p=2, dim=1, eps=1e-6)
        
        # REVERT TO SIGMOID: Preserves the strict (0, 2) bound required for valid Householder reflections
        beta = 2.0 * torch.sigmoid(self.beta_head(ctx)) 
        
        v = self.v_head(ctx) 
        
        # 4. Exact matrix-valued Delta Update
        X_mat = x_flat.view(B, C, self.dv, D, H, W)
        
        k_mat = k.unsqueeze(2)       
        v_mat = v.unsqueeze(1)       
        beta_mat = beta.unsqueeze(2) 
        
        k_T_X = torch.sum(k_mat * X_mat, dim=1, keepdim=True)
        update = beta_mat * k_mat * (v_mat - k_T_X)
        X_updated = X_mat + update 
        
        # -----------------------------------------------------------
        # 5. CONSTRAINED CC COMPRESSION
        # -----------------------------------------------------------
        X_transposed = X_updated.permute(0, 3, 4, 5, 1, 2)
        X_reshaped = X_transposed.reshape(B * D * H * W, C, self.dv)
        
        # Explicit L2 Weight Normalization
        # Normalizes the convolution weights across the dv dimension (dim=2)
        # Physically blocks the network from arbitrarily inflating parameters to memorize large organs
        norm_cc_weight = F.normalize(self.compress_cc.weight, p=2, dim=2)
        
        # Apply Channel Convolution using the strictly bounded weights
        out_cc = F.conv1d(X_reshaped, norm_cc_weight, groups=self.channels)
        
        # Unfold back to 3D spatial format: (B, C, D, H, W)
        out = out_cc.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        return out, k, beta, update

# CRITICAL ALIAS: Restores compatibility with UMambaBot_3d.py imports
DualBranchDDL = DDL_CC_3D