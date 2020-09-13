import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Todo make box plot with several inputs
def box_plot(df, feat):
    sns.boxplot(df[feat])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feat, fontsize=12)
    plt.show()

def scatter_plot(df, target, feat):
    sns.scatterplot(df[target], df[feat])
    plt.ylabel(feat, fontsize=12)
    plt.xlabel(target, fontsize=12)
    plt.show()

def dist_plot(df, feat):
    sns.distplot(df[feat])
    plt.show()