import torch

from .physics import Kart, Puck


class Environment:
    def __init__(self, batch_size=32, dt=0.1, max_frame=1200, device='cpu'):
        self.batch_size = batch_size
        self.dt = dt
        self.frame = 0
        self.max_frame = max_frame
        self.device = device
        self.kart = Kart(batch_size, device=device)
        self.puck = Puck(batch_size, device=device)

    def new_match(self):
        self.kart.position = torch.tensor((0.0, 50.0), device=self.device)
        self.kart.position += 10 * (torch.rand((self.batch_size, 2), device=self.device) - 0.5)
        self.kart.velocity = torch.zeros(self.batch_size, device=self.device)
        self.kart.rotation = -0.5 * torch.pi * torch.ones(self.batch_size, device=self.device)
        self.puck.position = torch.zeros((self.batch_size, 2), device=self.device)
        self.puck.velocity = torch.zeros(self.batch_size, device=self.device)
        self.puck.rotation = torch.zeros(self.batch_size, device=self.device)
        self.frame = 0

    def update(self):
        self.kart.update(self.dt)
        self.puck.update(self.dt)
        self.frame += 1

    @property
    def done(self):
        return self.frame >= self.max_frame
