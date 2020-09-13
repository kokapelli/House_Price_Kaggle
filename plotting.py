import matplotlib.pyplot as plt
import pandas as pd

"""
    Desc: Returns the frequency distribution of the columns values
    Param1: Dataset
    Param2: Column name
    Output: Frequency distribution of the input dataset and column name
"""
def plot_freq_dist(df, feat):
    column_count = df[feat].value_counts()
    plt.bar(column_count.index, column_count.values, alpha=0.9)
    plt.title('Frequency Distribution of ' + feat)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feat, fontsize=12)
    plt.show()

def box_plot(df, feat):
    plt.boxplot(df[feat])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feat, fontsize=12)
    plt.show()

def scatter_plot(df, target, feat):
    plt.scatter(df[target], df[feat])
    plt.ylabel(feat, fontsize=12)
    plt.xlabel(target, fontsize=12)
    plt.show()