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
    plt.show()

def plot_xyz(selection, title=''):
    colors = {1:'red', 2:'green', 3:'blue', 4:'yellow'}
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(title)
    ax.set_xlabel('x acc')
    ax.set_ylabel('y acc')
    ax.set_zlabel('z acc')
    ax.scatter(selection.x, selection.y, selection.z, c=selection.label.map(colors));

