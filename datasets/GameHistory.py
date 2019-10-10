import torch
from torch.utils import data
from collections import namedtuple


class GameHistory(data.Dataset):
    def __init__(self, df, bs, state_size, num_actions, device=None):
        assert df.columns == ['state', 'action', 'next_state', 'reward'], 'Data not in expected format.'

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.bs = bs
        self.df = df
        self.state_size = state_size
        self.num_actions = num_actions

    def preprocess_data(self, data):
        return self.transition(*zip(*data))

    def __len__(self):
        return len(self.df) // self.bs + 1

    def __getitem__(self, index):
        return self.preprocess_data(self.df.iloc[index])
