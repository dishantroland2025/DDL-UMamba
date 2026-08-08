import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralPhysicsLoss(nn.Module):
    def __init__(self, lambda_sparsity=0.05, lambda_boundary=0.5):
        """
        Forces the DDL block to learn interpretable geometric filters.
        Multi-Scale & Multi-Class safe version.
        """
        super().__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_boundary = lambda_boundary

    def forward(self, beta_maps_list, target_mask):
        """
        Args:
            beta_maps_list: List of beta maps [B, 1, D, H, W] from encoder stages.
            target_mask: Ground Truth segmentation (can be a list if deep supervision is on).
        """
        total_physics_loss = 0.0
        
        # Handle nnU-Net deep supervision (target might be a list of scales)
        # We only need the highest resolution target to extract pristine boundaries.
        if isinstance(target_mask, list):
            primary_target = target_mask[0].float()
        else:
            primary_target = target_mask.float()
            
        for beta_map in beta_maps_list:
            b_shape = beta_map.shape[2:] # (D, H, W)
            
            # 1. Downsample target to match this encoder stage's resolution.
            # CRITICAL: Must use 'nearest' to preserve discrete integer class labels!
            target_resized = F.interpolate(primary_target, size=b_shape, mode='nearest')
            
            # 2. Sparsity Loss
            loss_sparsity = torch.mean(torch.abs(beta_map))

            # 3. Multi-Class Boundary Detection (Max-Min Pooling trick)
            with torch.no_grad():
                if beta_map.dim() == 5: # 3D
                    max_val = F.max_pool3d(target_resized, kernel_size=3, stride=1, padding=1)
                    # PyTorch has no min_pool, so we use max_pool on the negative tensor
                    min_val = -F.max_pool3d(-target_resized, kernel_size=3, stride=1, padding=1)
                else: # 2D fallback
                    max_val = F.max_pool2d(target_resized, kernel_size=3, stride=1, padding=1)
                    min_val = -F.max_pool2d(-target_resized, kernel_size=3, stride=1, padding=1)
                
                # An edge exists wherever the max class in a local window differs from the min class.
                # This perfectly isolates RV/MYO, MYO/LV, and MYO/Background boundaries.
                edges = (max_val != min_val).float()

            # 4. Boundary Loss
            loss_boundary = F.mse_loss(beta_map, edges)
            
            total_physics_loss += (self.lambda_sparsity * loss_sparsity) + (self.lambda_boundary * loss_boundary)

        # Average across scales to keep the gradient magnitude stable
        return total_physics_loss / len(beta_maps_list)