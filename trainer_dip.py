from trainer_base import *

class TrainerDIP(TrainerBase):
    loss_names = ["rec"]
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        noise_reg_std: float,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        img_0: Union[torch.Tensor, None] = None,
        nsteps: int = 100,
        show_every: int = 0,
        seed: int = 999,
        device: str = "cpu",
    ):
        super().__init__(img_1=img_1, img_2=img_2, img_0=img_0, nsteps=nsteps, show_every=show_every, seed=seed, device=device)
        self.net = net.to(device)
        self.optimizer = optimizer
        self.noise_reg_std = noise_reg_std
        self.prior = torch.randn(*self.lin_0.shape).to(self.lin_0)
        
    def train_step(self):
        self.net.train()
        self.optimizer.zero_grad()
        prior = add_noise(self.prior, self.noise_reg_std)
        rec = self.net(prior[None])[0]
        rec_loss = (self.rec_loss_1(rec) + self.rec_loss_2(rec)) / 2
        rec_loss.backward()
        self.optimizer.step()
        return rec_loss.item(),
    
    @torch.no_grad()
    def reconstruct(self):
        self.net.eval()
        rec = self.net(self.prior[None])[0]
        return rec