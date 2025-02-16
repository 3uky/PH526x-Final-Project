from data_presentation import *
import numpy as np

def similarity(predictions, Y_val):
    return 100 * np.mean(predictions == Y_val)

def analyze_result(df_test, df_test_labels):
    plot_activity(df_test.timestamp, df_test.label)
    plot_activity(df_test_labels.timestamp, df_test_labels.label_knn_1, "knn neightbors=1 (without smoothing)")
    plot_activity(df_test_labels.timestamp, df_test_labels.label, "knn neightbors=10 (final result)")
    # print("Similarity between knn 1 and 10 is {}%".format(similarity(df_test_labels.label_knn_1, df_test_labels.label)))

    result_labels = df_test_labels.label.tolist()
    print(f"Final classified labels: {result_labels}")