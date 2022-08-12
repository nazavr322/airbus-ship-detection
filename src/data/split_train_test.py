import os
from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

from src import ROOT_DIR


load_dotenv()  # load environment variables from .env file
RANDOM_STATE = int(os.environ['RANDOM_STATE'])


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('data_path', help='path to the .csv file')
    parser.add_argument(
        'out_path', help='path to the folder to store train and test files'
    )
    test_size_help_msg = """should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the test split."""
    parser.add_argument('test_size', nargs='?', type=float, default=.15, 
                        help=test_size_help_msg)
    return parser


def split_train_test(
    data_path: str, test_size: int | float
) -> tuple[pd.DataFrame, ...]:
    """Splits data into train and test""" 
    raw_df = pd.read_csv(os.path.join(ROOT_DIR, data_path))
    train_df, test_df = train_test_split(
        raw_df, test_size=test_size, random_state=RANDOM_STATE
    ) 
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)
    return train_df, test_df


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    train_df, test_df = split_train_test(args.data_path, args.test_size)

    # construct full path to a target folder
    target_folder = os.path.join(ROOT_DIR, args.out_path)
    # create target folder if it's doesn't exist
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    # save dataframes as .csv files
    train_df.to_csv(
        os.path.join(target_folder, 'train_data.csv'), index=False
    )
    test_df.to_csv(os.path.join(target_folder, 'test_data.csv'), index=False)
