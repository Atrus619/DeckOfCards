import torch
from torch.utils import data


class GameHistory(data.Dataset):
    def __init__(self, df, batch_size, state_size, num_actions, device=None):
        assert all(df.columns == ['state', 'action', 'next_state', 'reward']), 'Data not in expected format.'

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.batch_size = batch_size
        self.df = df
        self.state_size = state_size
        self.num_actions = num_actions

        self.states = self.preprocess_series(series=self.df.state, dtype=torch.float32)
        self.actions = self.preprocess_series(series=self.df.action, dtype=torch.int64)
        self.next_states = self.preprocess_series(series=self.df.next_state, dtype=torch.float32)
        self.rewards = torch.tensor(self.df.reward.astype(float), device=self.device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.next_states[index], self.rewards[index]

    def preprocess_series(self, series, dtype):
        # dtype should be a torch dtype
        series = series.str.split(',').apply(lambda x: list(map(float, x)))
        return torch.tensor(series, device=self.device, dtype=dtype)
