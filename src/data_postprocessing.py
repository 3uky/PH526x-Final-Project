from sklearn.neighbors import KNeighborsClassifier

def smooth_knn(df_train, df_test, n_neigh=50):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(df_train.timestamp.values.reshape(-1, 1), df_train.label)
    return neigh.predict(df_test.timestamp.values.reshape(-1, 1))

# smoothing result with knn (10 becouse there are 10 samples in test_time_series.csv on 1 sample in test_labels.csv)
def postprocessing(df_test, df_test_labels):
    df_test_labels["label"] = smooth_knn(df_test, df_test_labels, n_neigh=10)
    df_test_labels["label_knn_1"] = smooth_knn(df_test, df_test_labels, n_neigh=1)

