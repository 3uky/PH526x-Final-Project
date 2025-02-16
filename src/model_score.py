from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from siml.sk_utils import *
from siml.signal_analysis_utils import *
import numpy as np
import warnings

warnings.filterwarnings("ignore") # ignore future warning

def generate_validation_data(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8, random_state=1)
    forest_classifier = RandomForestClassifier(max_depth=4, random_state=0)
    forest_classifier.fit(X_train, Y_train)
    Y_pred = forest_classifier.predict(X_val)
    return Y_val, Y_pred

def evaluate_confusion_matrix(X, Y):
    Y_val, Y_pred = generate_validation_data(X, Y)
    matrix = confusion_matrix(Y_val, Y_pred, normalize="true")

    print(matrix)
    print(f"Average diagonal precision after training data resampling: {(np.trace(matrix) / 4 * 100):.2f}%")

def model_accuracy(estimator, X, y):
    estimator.fit(X, y)
    predictions = estimator.predict(X)
    return accuracy_score(y, predictions)

def score_classifier(classifier, X, Y):
    score = cross_val_score(classifier, X, Y, cv=10, scoring=model_accuracy)
    print(classifier)
    print("mean score:", np.mean(score))

def score_models(X, Y):
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
