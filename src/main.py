import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import *
from data_presentation import *
from data_preprocessing import *
from data_postprocessing import *
from model_score import *

def read_data():
    df_labels = pd.read_csv(TRAIN_LABELS)
    df_train = pd.read_csv(TRAIN_TIME_SERIES)
    df_test = pd.read_csv(TEST_TIME_SERIES)
    df_test_labels = pd.read_csv(TEST_LABELS)

    merge_by_label(df_train, df_labels)
    calculate_magnitude_covariate(df_test, df_train)

    return df_train, df_test, df_test_labels

def store_result(df_test_labels):
    df_test_labels.to_csv(TEST_LABELS)

def main():
    df_train, df_test, df_test_labels = read_data()

    print_input_data(df_train)

    X, Y = data_preprocessing(df_train)

    # score
    evaluate_confusion_matrix(X, Y)  # evaluation of train data re-balancing
    score_models(X, Y)  # evaluation of different model classifiers

    # train model
    model = RandomForestClassifier(max_depth=4, random_state=0)
    model.fit(X, Y)

    # classification
    df_test["label"] = model.predict(df_test[all_covarites])

    postprocessing(df_test, df_test_labels)

    print_result(df_test, df_test_labels)

    store_result(df_test_labels)

if __name__=="__main__":
    main()