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
        'out_path', help='path to the folder to store train and validation files'
    )
    parser.add_argument('--val_size', nargs='?', type=int, default=2000, 
                        help='represents the absolute number of validation samples')
    return parser


def split_train_val(
    data_path: str, val_size: int | float, random_state: int = 1
) -> tuple[pd.DataFrame, ...]:
    """Splits data into train and validation""" 
    raw_df = pd.read_csv(os.path.join(ROOT_DIR, data_path))
    train_df, val_df = train_test_split(
        raw_df,
        test_size=val_size,
        random_state=random_state,
        stratify=raw_df.ShipCount
    ) 
    train_df.reset_index(inplace=True, drop=True)
    val_df.reset_index(inplace=True, drop=True)
    return train_df, val_df


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    # perform train/val split
    train_df, val_df = split_train_val(args.data_path, args.val_size, RANDOM_STATE)

    # construct full path to a target folder
    target_folder = os.path.join(ROOT_DIR, args.out_path)
    # create target folder if it's doesn't exist
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    # save dataframes as .csv files
    train_df.to_csv(
        os.path.join(target_folder, 'train_data.csv'), index=False
    )
    val_df.to_csv(os.path.join(target_folder, 'val_data.csv'), index=False)
