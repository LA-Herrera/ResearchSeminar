import torch
from collections import OrderedDict
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.logger import get_root_logger
from basicsr.losses import build_loss

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

    def init_training_settings(self) -> None:
        super().init_training_settings()

        train_opt = self.opt['train']

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None
    
    def optimize_parameters(self, current_iter: int) -> None:
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total   = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)
