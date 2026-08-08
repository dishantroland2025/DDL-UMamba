from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn, autocast
import torch
from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

from nnunetv2.training.loss.spectral_loss import SpectralPhysicsLoss

class nnUNetTrainerUMambaEnc(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # Initialize the Multi-Scale DDL Loss and move to device
        self.ddl_loss_func = SpectralPhysicsLoss(lambda_sparsity=0.05, lambda_boundary=0.5).to(self.device)

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager, dataset_json, configuration_manager: ConfigurationManager, num_input_channels, enable_deep_supervision: bool = True) -> nn.Module:
        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_enc_2d_from_plans(plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = get_umamba_enc_3d_from_plans(plans_manager, dataset_json, configuration_manager, num_input_channels, deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")
        
        # UPDATED: Prevent log confusion by reflecting the correct dv capacity
        print("UMambaEnc DDL-CC (dv=4, Constrained) Architecture Built.")
        return model

    def train_step(self, batch: dict):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # OPTIMIZATION: set_to_none=True clears memory faster
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True):
            # Forward Pass
            output = self.network(data)
            
            # Unpack the tuple: (logits, list_of_beta_maps)
            if isinstance(output, tuple):
                logits, beta_maps_list = output
            else:
                logits = output
                beta_maps_list = None

            # Calculate Standard nnU-Net Loss
            l = self.loss(logits, target)

            # Add Multi-Scale Spectral Physics Loss
            if beta_maps_list is not None:
                l_physics = self.ddl_loss_func(beta_maps_list, target)
                total_loss = l + (0.1 * l_physics)
            else:
                total_loss = l

        # Backward Pass & Optimization
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': total_loss.detach().cpu().numpy()}