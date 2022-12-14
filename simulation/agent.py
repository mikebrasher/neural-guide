import torch


class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_actions = 3

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observations.size(0), self.num_actions))
