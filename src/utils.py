import numpy as np
import pandas as pd


def get_unique(train_data):
    unique_elems = {}
    for col in train_data.columns:
        unique_elems[col] = train_data[col].unique()
    return unique_elems
