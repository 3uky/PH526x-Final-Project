import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from siml.sk_utils import *
from sklearn.neighbors import KNeighborsClassifier
import warnings
import time

from config import *

def add_magnitude(df):
    df['m'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)

# read labels from train_labels.csv and set label value in train_time_series data frame based on nearest timestamp value
def add_label(df_train):
    df_train['label'] = df_train['timestamp'].map(lambda timestamp: get_label(df_labels, timestamp))

def get_label(df_labels, timestamp):
    timestamps = np.asarray(df_labels.timestamp)
    idx = (np.abs(timestamps - timestamp)).argmin()
    return df_labels.iloc[idx].label

def smooth_knn(df_train, df_test, n_neigh=50):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(df_train.timestamp.values.reshape(-1, 1), df_train.label)
    return neigh.predict(df_test.timestamp.values.reshape(-1, 1))

warnings.filterwarnings('ignore')

start_time = time.process_time()

df_labels = pd.read_csv(TRAIN_LABELS, usecols=['timestamp', 'label'])
df_train = pd.read_csv(TRAIN_TIME_SERIES, usecols=['timestamp', 'x', 'y', 'z'])

add_magnitude(df_train)
add_label(df_train)

Y = df_train['label']
X = df_train[['x', 'y', 'z', 'm']]

ros = RandomOverSampler()
X, Y = ros.fit_resample(X, Y)

forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
forest_classifier.fit(X, Y)

df_test = pd.read_csv(TEST_TIME_SERIES, usecols=['timestamp', 'x', 'y', 'z'])
add_magnitude(df_test)
test_covariates = df_test[['x', 'y', 'z', 'm']]
df_test['label'] = forest_classifier.predict(test_covariates)

df_test_labels = pd.read_csv(TEST_LABELS, usecols=['timestamp', 'label'])
df_test_labels['label'] = smooth_knn(df_test, df_test_labels, n_neigh=10)

df_test_labels.to_csv(TEST_LABELS)

end_time = time.process_time()
print(f"{end_time - start_time} s")