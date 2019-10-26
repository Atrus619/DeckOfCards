from config import Config as cfg
import pandas as pd
import os


def get_experiment_file(file):
    return pd.read_csv(os.path.join(cfg.experiment_folder, file))


