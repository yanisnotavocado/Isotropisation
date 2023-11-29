from trainer_base import *

class TrainerPair(TrainerBase):
    loss_names = ["rec_1", "rec_2", "sim"]
    def __init__(
        self,
        net_1: nn.Module,
        net_2: nn.Module,
        optimizer: torch.optim.Optimizer,
        sim_weight: Union[float, None],
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        img_0: Union[torch.Tensor, None] = None,
        nsteps: int = 100,
        show_every: int = 0,
        seed: int = 999,
        device: str = "cpu",
    ):
        super().__init__(img_1=img_1, img_2=img_2, img_0=img_0, nsteps=nsteps, show_every=show_every, seed=seed, device=device)
        self.net_1 = net_1.to(device)
        self.net_2 = net_2.to(device)
        self.optimizer = optimizer
        self.sim_weight = sim_weight
        
    def train_step(self):
        self.net_1.train()
        self.net_2.train()

        rec_1 = self.net_1(self.lin_2[None])[0]
        rec_loss_1 = self.rec_loss_1(rec_1)
        rec_2 = self.net_2(self.lin_1[None])[0]
        rec_loss_2 = self.rec_loss_2(rec_2)
        sim_loss = self.criterion(rec_1, rec_2)
        loss = rec_loss_1 + rec_loss_2
        if self.sim_weight:
            loss = loss + self.sim_weight * sim_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return rec_loss_1.item(), rec_loss_2.item(), sim_loss.item()
    
    @torch.no_grad()
    def reconstruct(self):
        self.net_1.eval()
        self.net_2.eval()
        rec_1 = self.net_1(self.lin_2[None])[0]
        rec_2 = self.net_2(self.lin_1[None])[0]
        rec = (rec_1 + rec_2) / 2
        return rec