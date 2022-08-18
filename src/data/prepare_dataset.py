import os
from argparse import ArgumentParser

import pandas as pd

from src import ROOT_DIR
from ..features import add_ship_count


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument(
        'raw_data_path', help='path to .csv file with raw data'
    )
    parser.add_argument('out_path', help='path to an output .csv file')
    return parser


def _join_rle_strings(row):
    """Joins multiple rle strings in one"""
    try:
        return ' '.join(row)
    except TypeError:
        return row


# The dataset is built in such a way that each row in the table corresponds to
# one ship mask (e.g imagine picture containing 3 ships => there will be 3 rows
# with the same filename but different rle string in a table).
# Idea of this function is to remove duplicated rows, but keep all the masks.
# It can be achieved by concatenating all corresponding masks for each picture.


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    For each image, concatenates all corresponding masks and deletes its copies
    """
    data['EncodedPixels'] = data.groupby('ImageId')['EncodedPixels'].transform(
        _join_rle_strings
    )
    return data.drop_duplicates('ImageId', ignore_index=True)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # parse cmd arguments

    raw_df = pd.read_csv(os.path.join(ROOT_DIR, args.raw_data_path))
    # add column with ship counts
    df_with_counts = add_ship_count(raw_df)
    # remove duplicates
    cleaned_df = remove_duplicates(df_with_counts)

    # create target folders if they don't exist
    full_out_path = os.path.join(ROOT_DIR, args.out_path)
    full_target_dir_path = os.path.dirname(full_out_path)
    if not os.path.isdir(full_target_dir_path):
        os.makedirs(full_target_dir_path)

    # save cleaned dataset
    cleaned_df.to_csv(full_out_path, index=False)
