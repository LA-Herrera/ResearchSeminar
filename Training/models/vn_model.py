import torch
from collections import OrderedDict
from BasicSR.basicsr.models.sr_model import SRModel
from BasicSR.basicsr.utils.registry import MODEL_REGISTRY
from BasicSR.basicsr.utils.logger import get_root_logger

@MODEL_REGISTRY.register()
class VN_Model(SRModel):
    
    def setup_optimizers(self) -> None:
        train_opt = self.opt['train']
        optim_params = []

        for name, param in self.net_g.named_parameters():
            if param.requires_grad:
                optim_params.append(param)
            else:
                logger = get_root_logger()
                logger.warning(f"Parameter {name} is not being optimized")
        
        self.optimizer_g = torch.optim.Adam(
            optim_params,
            lr=train_opt['optim_g']['lr'],
            betas=(0.9, 0.99),
            weight_decay=train_opt['optim_g'].get('weight_decay', 0)
        )
        self.optimizers.append(self.optimizer_g)