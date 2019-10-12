import torch
from torch.utils import data


class GameHistory(data.Dataset):
    def __init__(self, df, bs, state_size, num_actions, device=None):
        assert all(df.columns == ['state', 'action', 'next_state', 'reward']), 'Data not in expected format.'

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.bs = bs
        self.df = df
        self.state_size = state_size
        self.num_actions = num_actions

        self.states = self.preprocess_series(self.df.state)
        self.actions = self.preprocess_series(self.df.action)
        self.next_states = self.preprocess_series(self.df.next_state)
        self.rewards = torch.tensor(self.df.reward.astype(float), device=self.device)

    # TODO: I don't think we are gonna use dis
    def preprocess_data_MAYBE_USE_THIS_LATA(self, data):
        return self.transition(*zip(*data))

    def __len__(self):
        return len(self.df) // self.bs + 1

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.next_states[index], self.rewards[index]

    def preprocess_series(self, series):
        series = series.str.split(',').apply(lambda x: list(map(float, x)))
        return torch.tensor(series, device=self.device)
