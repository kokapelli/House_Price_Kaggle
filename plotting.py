import matplotlib.pyplot as plt

"""
    Desc: Returns the frequency distribution of the columns values
    Param1: Dataset
    Param2: Column name
    Output: Frequency distribution of the input dataset and column name
"""
def plot_freq_dist(df, column):
    fig = plt.figure()
    column_count = df[column].value_counts()
    plt.bar(column_count.index, column_count.values, alpha=0.9)
    plt.title('Frequency Distribution of ' + column)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.show()