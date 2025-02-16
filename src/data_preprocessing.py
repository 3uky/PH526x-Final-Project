import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from siml.signal_analysis_utils import *

def add_magnitude(df):
    df['m'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)

# read labels from train_labels.csv and set label value in train_time_series data frame based on nearest timestamp value
def add_label(df_train, df_labels):
    df_train['label'] = df_train['timestamp'].map(lambda timestamp: get_label(df_labels, timestamp))

def get_label(df_labels, timestamp):
    timestamps = np.asarray(df_labels.timestamp)
    idx = (np.abs(timestamps - timestamp)).argmin()
    return df_labels.iloc[idx].label

def resampling(X, Y):
    print("Imbalanced class distribution before re-sampling:\n{}".format(Y.value_counts()))
    ros = RandomOverSampler()
    X, Y = ros.fit_resample(X, Y)
    print("Class distribution after re-sampling:\n{}".format(Y.value_counts()))
    return X, Y