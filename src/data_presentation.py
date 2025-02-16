import matplotlib.pyplot as plt
import numpy as np

def plot_xy(x, y, y_title="acceleration"):
    fig = plt.figure(figsize=(10, 3))
    plt.xlabel("timestamp")
    plt.ylabel(y_title)
    plt.plot(x, y)
    plt.show()

def plot_activity(x, y, title=''):
    fig = plt.figure(figsize=(10, 3))
    plt.xlabel("timestamp")
    plt.ylabel("activity")
    plt.plot(x, y)
    # plt.yticks([1, 2, 3, 4], ["standing", "walking", "stairs down", "stairs up"])
    plt.yticks([1, 2, 3, 4])
    plt.title(title)
    plt.show()

def plot_data(df):
    selection = df.iloc[:]
    plt.figure(figsize=(10, 3))
    plt.plot(selection.timestamp, selection.x, linewidth=0.5, color='r', label='x acc')
    plt.plot(selection.timestamp, selection.y, linewidth=0.5, color='b', label='y acc')
    plt.plot(selection.timestamp, selection.z, linewidth=0.5, color='g', label='z acc')
    plt.xlabel('timestamp')
    plt.ylabel('acceleration')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    plt.show

def accuracy(predictions, Y_val):
    return 100 * np.mean(predictions == Y_val)

def print_input_data(df_train):
    #plot_activity(df_labels.timestamp, df_labels.label) # labels/activities are consistent in time
    plot_activity(df_train.timestamp, df_train.label)
    plot_data(df_train)
    plot_xy(df_train.timestamp, df_train.m, "magnitude")

def print_result(df_test, df_test_labels):
    plot_activity(df_test.timestamp, df_test.label)
    plot_activity(df_test_labels.timestamp, df_test_labels.label_knn_1, "knn neightbors=1 (without smoothing)")
    plot_activity(df_test_labels.timestamp, df_test_labels.label, "knn neightbors=10 (final result)")
    # print("Similarity between knn 1 and 10 is {}%".format(accuracy(df_test_labels.label_knn_1, df_test_labels.label)))

    result_labels = df_test_labels.label.tolist()
    print(result_labels)