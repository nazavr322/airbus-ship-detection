import os
from argparse import ArgumentParser

import pandas as pd
from dotenv import load_dotenv

from src import ROOT_DIR


load_dotenv()  # load environment variables from .env file
RANDOM_STATE = int(os.environ['RANDOM_STATE'])


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('input_path', help='path to .csv file')
    parser.add_argument('output_path', help='path to .csv file')
    parser.add_argument(
        '--samples_per_group',
        nargs='?',
        type=int,
        default=3000,
        help='maximum number of samples of a single class',
    )
    return parser


def _get_n_samples(group, n: int, random_state: int = 1):
    """
    Returns n random samples from group. If group has less values that n,
    returns the whole group.
    """
    if len(group) <= n:
        return group
    return group.sample(n, random_state=random_state)


def get_balanced_subset(
    df: pd.DataFrame, samples_per_group: int, random_state: int = 1
) -> pd.DataFrame:
    """
    Returns a dataframe where each class occurs at most samples_per_group times
    """
    args = (samples_per_group, random_state)
    balanced_df = df.groupby('ShipCount', as_index=False).apply(
        _get_n_samples, *args
    )
    return balanced_df.reset_index(drop=True)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    df = pd.read_csv(os.path.join(ROOT_DIR, args.input_path))

    # get more balanced subset
    balanced_df = get_balanced_subset(df, args.samples_per_group, RANDOM_STATE)

    # create target folders if they don't exist
    full_out_path = os.path.join(ROOT_DIR, args.output_path)
    full_target_dir_path = os.path.dirname(full_out_path)
    if not os.path.isdir(full_target_dir_path):
        os.makedirs(full_target_dir_path)

    # save balanced dataset
    balanced_df.to_csv(full_out_path, index=False)
