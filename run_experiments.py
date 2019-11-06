from config import Config
from copy import deepcopy
from pipeline.experiment import run_full_experiment
import argparse
import util.util as util

parser = argparse.ArgumentParser(description='Enter the filename of the  experiment file.')
parser.add_argument('filename', type=str, help='filename of experiment file')
args = parser.parse_args()


experiments = util.get_experiment_file(file=args.filename)

for i, experiment in experiments.iterrows():
    config = util.overwrite_cfg(exp=experiment, config=Config())
    run_full_experiment(config=config)
