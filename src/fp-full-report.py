import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

import warnings

from data_presentation import *
from config import *

warnings.filterwarnings("ignore")

df_labels = pd.read_csv(TRAIN_LABELS)
df_train = pd.read_csv(TRAIN_TIME_SERIES)

# PREPROCESSING
def add_magnitude(df):
    df['m'] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)

# read labels from train_labels.csv and set label value in train_time_series data frame based on nearest timestamp value
def add_label(df_train):
    df_train['label'] = df_train['timestamp'].map(lambda timestamp: get_label(df_labels, timestamp))


def get_label(df_labels, timestamp):
    timestamps = np.asarray(df_labels.timestamp)
    idx = (np.abs(timestamps - timestamp)).argmin()
    return df_labels.iloc[idx].label

# calculate new columns
add_magnitude(df_train)
add_label(df_train)

# PRESENTATION
plot_activity(df_labels.timestamp, df_labels.label) # labels/activities are consistent in time
plot_data(df_train)
plot_xy(df_train.timestamp, df_train.m, "magnitude")
plot_activity(df_train.timestamp, df_train.label)



# Specify classification_outcome Y and covariates X
classification_target = 'label'
all_covariates = ['x', 'y', 'z', 'm']
Y = df_train[classification_target]
X = df_train[all_covariates]

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from siml.signal_analysis_utils import *

print("Imbalanced class distribution before re-sampling:\n{}".format(Y.value_counts()))

ros = RandomOverSampler()
X, Y = ros.fit_resample(X, Y)

print("Class distribution after re-sampling:\n{}".format(Y.value_counts()))

from sklearn.metrics import confusion_matrix


def generate_validation_data(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8, random_state=1)
    forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
    forest_classifier.fit(X_train, Y_train)
    Y_pred = forest_classifier.predict(X_val)
    return Y_val, Y_pred


def evaluate_confusion_matrix(X, Y):
    Y_val, Y_pred = generate_validation_data(X, Y)
    cm = confusion_matrix(Y_val, Y_pred, normalize="true")

    print(cm)
    print("Average diagonal precision {:.2f}%".format(np.trace(cm) / 4 * 100))


evaluate_confusion_matrix(X, Y)

from siml.sk_utils import *
from siml.signal_analysis_utils import *
import sensormotion as sm
from scipy.fftpack import fft


def model_accuracy(estimator, X, y):
    estimator.fit(X, y)
    predictions = estimator.predict(X)
    return accuracy_score(y, predictions)


def score_classifier(classifier, X, Y):
    score = cross_val_score(classifier, X, Y, cv=10, scoring=model_accuracy)
    print(classifier)
    print("mean score:", np.mean(score))


# instanciate classifers
logistic_regression = LogisticRegression()
forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)

# evaluate a score by cross-validation
score_classifier(logistic_regression, X, Y)
score_classifier(forest_classifier, X, Y)

# split train data on training and testing in ration 8:2
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8, random_state=1)

# check score for different models
models = batch_classify(X_train, Y_train, X_val, Y_val)
display_dict_models(models)

forest_classifier.fit(X, Y)

# load test data
df_test = pd.read_csv(TEST_TIME_SERIES)

# data preprocessing
add_magnitude(df_test)

# classification
test_covariates = df_test[all_covariates]
df_test["label"] = forest_classifier.predict(test_covariates)

# check result in plot
plot_activity(df_test.timestamp, df_test.label)

from sklearn.neighbors import KNeighborsClassifier


def smooth_knn(df_train, df_test, n_neigh=50):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(df_train.timestamp.values.reshape(-1, 1), df_train.label)
    return neigh.predict(df_test.timestamp.values.reshape(-1, 1))


# load test_labels
df_test_labels = pd.read_csv(TEST_LABELS)

# smoothing result with knn (10 becouse there are 10 samples in test_time_series.csv on 1 sample in test_labels.csv)
df_test_labels["label"] = smooth_knn(df_test, df_test_labels, n_neigh=10)
df_test_labels["label_knn_1"] = smooth_knn(df_test, df_test_labels, n_neigh=1)


def accuracy(predictions, Y_val):
    return 100 * np.mean(predictions == Y_val)


# print results
plot_activity(df_test_labels.timestamp, df_test_labels.label_knn_1, "knn neightbors=1 (without smoothing)")
plot_activity(df_test_labels.timestamp, df_test_labels.label, "knn neightbors=10 (final result)")
result_labels = df_test_labels.label.tolist()
# print("Similarity between knn 1 and 10 is {}%".format(accuracy(df_test_labels.label_knn_1, df_test_labels.label)))

print(result_labels)

# Store result into test_labels.csv
df_test_labels.to_csv(TEST_LABELS)