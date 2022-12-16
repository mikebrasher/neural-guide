import torch
# import torch.utils.tensorboard as tb

from argparse import ArgumentParser
from os import path
import time

from simulation.environment import Environment

seed = 1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ActionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # size_out = floor( (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1 )
        # size_out = size_in + 2 * padding - dilation * (kernel_size - 1)
        padding = dilation * (kernel_size - 1) // 2
        self.network = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, dilation=dilation, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, dilation=dilation, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                            padding=padding, dilation=dilation, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, dilation=dilation),
            torch.nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        y0 = self.network(x)
        y1 = self.residual(x)
        result = y0 + y1
        return result


class ActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_observations = 6
        hidden_size = 32
        num_action = 4
        self.actor_network = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(num_observations, 1, kernel_size=hidden_size),
            ActionBlock(1, hidden_size // 2, kernel_size=5, dilation=1),
            ActionBlock(hidden_size // 2, hidden_size, kernel_size=5, dilation=2),
        )
        self.actor_classifier = torch.nn.Linear(hidden_size, num_action)

        self.critic_linear = torch.nn.Linear(num_observations, hidden_size)
        self.critic_classifier = torch.nn.Linear(hidden_size, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if len(x.shape) == 1:
            value = self.critic_linear(x)
            value = self.relu(value)
            value = self.critic_classifier(value)

            y = x.unsqueeze(0).unsqueeze(2)
            policy = self.actor_network(y).squeeze(0)
            policy = policy.mean(dim=[1])
            policy = self.actor_classifier(policy)
        else:
            value = 0

            y = x.unsqueeze(2)
            policy = self.actor_network(y)
            policy = policy.mean(dim=[1])
            policy = self.actor_classifier(policy)

        return value, policy


def train_actor(arguments):
    torch.manual_seed(seed)

    environment = Environment(batch_size=1024*1024, device=device)

    train_start = time.time()
    environment.new_match()
    while not environment.done:
        environment.update()
    print('Trained in {} seconds'.format(time.time() - train_start))


def save_model(model, epoch=None):
    save_dir = path.dirname(path.abspath(__file__))
    if epoch is None:
        filename = 'actor.th'
    else:
        filename = 'actor.epoch{:03d}.th'.format(epoch)
    torch.save(model.state_dict(), path.join(save_dir, filename))


def load_model(filename):
    save_dir = path.dirname(path.abspath(__file__))
    state_dict = torch.load(path.join(save_dir, filename), map_location='cpu')
    model = ActorCritic()
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--logdir')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    train_actor(args)
