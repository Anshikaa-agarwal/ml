import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn has some inbuilt datasets

# 1st dataset: total bill vs total tip
# scatter graph
tips = sns.load_dataset('tips')
print(tips)
sns.scatterplot(x='total_bill', y='tip', data=tips, hue='sex', size='size', style='time')
plt.show()

# bar graph
sns.barplot(data=tips, x='day', y='tip')
plt.show()

# histogram
sns.histplot(tips['total_bill'], kde=True)  # KDE shows the data's probability density
plt.show()


# skilearn datasets
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

house = pd.DataFrame(housing.data, columns = housing.feature_names)
print(house.head())