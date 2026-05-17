import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=sns.load_dataset('iris')
num_col_1='sepal_length'
num_col_2='sepal_width'
plt.figure(figsize=(8,6))
sns.scatterplot(x=df[num_col_1], y=df[num_col_2], color='blue')
plt.title(f'Scatter plot between {num_col_1} and {num_col_2}')
plt.xlabel(num_col_1)
plt.ylabel(num_col_2)
plt.show()

pearson_corr=df[num_col_1].corr(df[num_col_2])
print(f'Pearson Coefficient between {num_col_1} and {num_col_2}: {pearson_corr}')
numeric_columns=df.select_dtypes(include=['number']).columns
print("\nNumeric Columns:")
print(numeric_columns)
cov_matrix=df[numeric_columns].cov()
print("\nCovariance Matrix:")
print(cov_matrix)
corr_matrix=df[numeric_columns].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
