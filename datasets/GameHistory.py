import torch
from torch.utils import data
from collections import namedtuple


class GameHistory(data.Dataset):
    def __init__(self, df, batch_size, state_size, num_actions, device=None):
        assert all(df.columns == ['state', 'action', 'meld_action', 'next_state', 'reward']), 'Data not in expected format.'

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.batch_size = batch_size
        self.state_size = state_size
        self.num_actions = num_actions

        states = self.preprocess_series(series=df.state, dtype=torch.float32)
        actions = self.preprocess_series(series=df.action, dtype=torch.int64)
        meld_actions = self.preprocess_series(series=df.meld_action, dtype=torch.int64)
        next_states = self.preprocess_series(series=df.next_state, dtype=torch.float32)
        rewards = torch.tensor(df.reward.astype(float), device=self.device)

        Data = namedtuple('Data', 'state action meld_action next_state reward')
        self.data = [Data(*x) for x in zip(states, actions, meld_actions, next_states, rewards)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]#self.data[index].state, self.data[index].action, self.data[index].meld_action, self.data[index].next_state, self.data[index].reward

    def preprocess_series(self, series, dtype):
        # dtype should be a torch dtype
        series = series.str.split(',').apply(lambda x: list(map(float, x)))
        return torch.tensor(series, device=self.device, dtype=dtype)
