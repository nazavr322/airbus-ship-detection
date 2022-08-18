import pandas as pd


def add_ship_count(data: pd.DataFrame) -> pd.DataFrame:
    """Adds a new column with count of ships on each picture"""
    data['ShipCount'] = data.groupby('ImageId')['ImageId'].transform('count')
    # at this point pictures without ships (EncodedPixels == NaN) also have
    # ShipCount == 1, the line below fixes it
    data.loc[data['EncodedPixels'].isnull(), 'ShipCount'] = 0
    return data
