from data_presentation import *

def analyze_input_data(df_train):
    #plot_activity(df_labels.timestamp, df_labels.label) # labels/activities are consistent in time
    plot_activity(df_train.timestamp, df_train.label, "train data - activities")
    plot_data(df_train)
    plot_xy(df_train.timestamp, df_train.m, "magnitude")

    # 3d plot for all activities colored by it's labels
    plot_xyz(df_train, "all activities")

    # 3d plot for each activity
    for label_number, activity in {1: 'standing', 2: 'walking', 3: 'stairs down', 4:'stairs up'}.items():
        plot_xyz(df_train[df_train.label == label_number], activity)