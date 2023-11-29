import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import interpolate1d, add_noise
from typing import Union, Sequence, Callable
from monai.utils import set_determinism
from IPython.display import clear_output
from tqdm import trange

class TrainerBase:
    loss_names: Sequence[str] = []
    def __init__(
        self,
        img_1: torch.Tensor,
        img_2: torch.Tensor,
        img_0: Union[torch.Tensor, None] = None,
        nsteps: int = 100,
        show_every: int = 0,
        seed: int = 999,
        device: str = "cpu",
    ):
        set_determinism(seed)
        
        self.img_1 = img_1.to(device)
        self.img_2 = img_2.to(device)
        self.img_0 = img_0
        self.nsteps = nsteps
        self.show_every = show_every
        self.seed = seed
        
        self.lin_1 = interpolate1d(self.img_1, self.img_2.size(1), 1)
        self.lin_2 = interpolate1d(self.img_2, self.img_1.size(2), 2)
        self.lin_0 = (self.lin_1 + self.lin_2) / 2
        self.min = self.lin_0.min()
        self.max = self.lin_0.max()
        assert self.lin_1.shape == self.lin_2.shape
        if img_0 is not None:
            self.img_0 = self.img_0.to(device)
            assert img_0.shape == self.lin_1.shape
        
        self.step = 0
        self.loss_b = 1e5
        self.rec_b = torch.zeros_like(self.lin_0)
        
        self.criterion = nn.L1Loss() #lambda x, y : F.mse_loss(x, y).sqrt()
        self.rec_loss_1 = lambda rec : self.criterion(interpolate1d(rec, self.img_1.size(1), 1), self.img_1)
        self.rec_loss_2 = lambda rec : self.criterion(interpolate1d(rec, self.img_2.size(2), 2), self.img_2)
        
    def train_step(self):
        raise NotImplementedError
    
    @torch.no_grad()
    def reconstruct(self):
        raise NotImplementedError
    
    def train(self):
        pbar = trange(self.step, self.nsteps)
        for i in pbar:
            losses = self.train_step()
            msg = ", ".join([f"{loss_name} = {loss:.3f}" for loss_name, loss in zip(self.loss_names, losses)])
            pbar.set_description(msg)
            
            rec_0 = self.reconstruct()
            loss_0 = (self.rec_loss_1(rec_0) + self.rec_loss_2(rec_0)) / 2
            if loss_0 < self.loss_b:
                self.loss_b = loss_0
                self.rec_b = rec_0
            if self.show_every and ((self.step + 1) % self.show_every == 0):
                self.show_progress()
            self.step += 1
            
    def show_progress(self):        
        clear_output(wait=True)
        plt.figure(figsize=(6, 3))
        for i, title, img in [(1, "Linear", self.lin_0), (2, "Proposed", self.rec_b)]:
            if self.img_0 is not None:
                title = title + f". Dist to Orig: {self.criterion(img, self.img_0):.3f}"
            img = ((img - self.min) / (self.max - self.min)).detach().cpu().permute(1, 2, 0).clip(0, 1)
            plt.subplot(1, 2, i) 
            plt.imshow(img, aspect="auto")
            plt.title(title, fontsize=10)
            plt.xticks([]), plt.yticks([])
        plt.show()