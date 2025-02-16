import os

DATA_PATH = "../data"
TRAIN_LABELS = os.path.join(DATA_PATH, "train_labels.csv")
TRAIN_TIME_SERIES = os.path.join(DATA_PATH, "train_time_series.csv")
TEST_TIME_SERIES = os.path.join(DATA_PATH, "test_time_series.csv")
TEST_LABELS = os.path.join(DATA_PATH, "test_labels.csv")

classification_target = 'label'
all_covariances = ['x', 'y', 'z', 'm']