"""
Implement various feature extraction techniques using tf and sklearn
"""
import os
import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.feature_column import (
    numeric_column, bucketized_column,
    categorical_column_with_vocabulary_list, embedding_column,
    crossed_column, indicator_column)
from tensorflow.keras.layers import DenseFeatures
