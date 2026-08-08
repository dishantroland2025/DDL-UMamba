from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn, autocast
import torch
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
# IMPORT YOUR NEW LOSS
from nnunetv2.training.loss.spectral_loss import SpectralPhysicsLoss

class nnUNetTrainerUMambaBot(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Initialize the DDL Loss
        self.ddl_loss_func = SpectralPhysicsLoss(lambda_sparsity=0.05, lambda_boundary=0.5)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        print("UMambaBot with DDL Architecture Built.")
        return model

    # ------------------------------------------------------------------
    # OVERRIDE THE TRAINING STEP TO HANDLE DDL OUTPUTS
    # ------------------------------------------------------------------
    def train_step(self, batch: dict):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Autocast for Mixed Precision (Standard in nnU-Net)
        with autocast(self.device.type, enabled=True):
            # 1. Run Forward Pass
            # DDL returns a tuple: (logits, k, beta)
            output = self.network(data)
            
            # Unpack the tuple
            if isinstance(output, tuple):
                logits, k_map, beta_map = output
            else:
                # Fallback if DDL is disabled or in eval mode
                logits = output
                beta_map = None

            # 2. Calculate Standard Dice/CE Loss
            l = self.loss(logits, target)

            # 3. Calculate Spectral Physics Loss (If beta exists)
            if beta_map is not None:
                l_physics = self.ddl_loss_func(beta_map, target)
                # Combine losses (0.1 weight for physics is usually safe)
                total_loss = l + (0.1 * l_physics)
            else:
                total_loss = l

        # 4. Backward Pass & Optimization
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': total_loss.detach().cpu().numpy()}