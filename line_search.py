import argparse
from itertools import product
from multiprocessing import Pool

import numpy as np

from experiments.train import main


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=False, default="experiments/configs/DQN_dyn.ini")
    parser.add_argument("--n-jobs", type=int, required=False, default=8)
    parser.add_argument("--n", type=int, required=False, default=9)
    parser.add_argument("--x-feat", type=str, required=False, default="queue")
    parser.add_argument("--y-feat", type=str, required=False, default="brake")
    args = parser.parse_args()

    p = Pool(args.n_jobs)
    ratios = np.linspace(0.0, 1.0, num=args.n)
    run_args = [Namespace(**{"config_path": args.config_path,
                             args.x_feat: str(x),
                             args.y_feat: str(1-x)}) for x in ratios]
    p.map(main, run_args)
