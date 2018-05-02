import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
train = pd.read_csv('Dataset/reg_train.csv', index_col=['instant', 'dteday'])
test = pd.read_csv('Dataset/reg_test.csv', index_col=['instant', 'dteday'])

# Visualize the distribution for each feature
for i in range(len(train.columns)):

    sns.distplot(train.iloc[:, i])
    plt.show()

# Find the correlations between each pair of features with pairplot
sns.pairplot(train)
plt.show()

# Find the correlations between each pair of features with heatmap
sns.heatmap(train.cov())
plt.show()
