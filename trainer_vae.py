from trainer_base import *
from models import VAE


class TrainerVAE(TrainerBase):
    loss_names = ["rec", "kld"]
    def __init__(
        self,
        vae: VAE,
        optimizer: torch.optim.Optimizer,
        beta: Union[Callable, float],
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        img_0: Union[torch.Tensor, None] = None,
        nsteps: int = 100,
        show_every: int = 0,
        seed: int = 999,
        device: str = "cpu",
    ):
        super().__init__(img_1=img_1, img_2=img_2, img_0=img_0, nsteps=nsteps, show_every=show_every, seed=seed, device=device)
        self.vae = vae.to(device)
        self.optimizer = optimizer
        self.beta = beta if callable(beta) else lambda step : beta
        self.inputs = torch.stack([self.lin_1, self.lin_2])
        
    def rec_loss(self, rec: torch.Tensor): return (self.rec_loss_1(rec[0]) + self.rec_loss_2(rec[1])) / 2
        
    def kld_loss(self, mu, logvar): return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
    def train_step(self):
        self.vae.train()
        self.optimizer.zero_grad()
        rec, mu, logvar, *_ = self.vae(self.inputs)
        rec_loss = self.rec_loss(rec)
        kld_loss = self.kld_loss(mu, logvar)
        beta = self.beta(self.step) * np.prod(mu.shape) / np.prod(self.inputs.shape)
        (rec_loss + beta * kld_loss).backward()
        self.optimizer.step()
        return rec_loss.item(), kld_loss.item()
    
    @torch.no_grad()
    def reconstruct(self):
        self.vae.eval()
        mu, logvar = self.vae.encode(self.inputs)
        z = self.vae.reparameterize(mu, logvar).mean(0, True)
        rec = self.vae.decode(z)[0]
        return rec
