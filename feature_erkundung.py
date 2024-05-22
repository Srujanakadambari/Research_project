# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import skew, pointbiserialr, shapiro
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# %%
# Load the dataset
dataset = pd.read_csv("/content/Solar_categorical.csv")
print(type(dataset))

# %%
# Check for missing values
missing_values = dataset.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values
dataset.fillna(dataset.mean(), inplace=True)

# %%
# Convert categorical columns to numerical
dataset['Weather'].replace(['Sunny', 'Cloudy'], [0, 1], inplace=True)
dataset['State'].replace(['Line-line', 'Open', 'Normal'], [0, 1, 2], inplace=True)

# %%
# Calculate skewness of numeric columns
numeric_cols = dataset.select_dtypes(include=np.number).columns
skewness = dataset[numeric_cols].apply(lambda x: x.skew())
print("Skewness of each numeric column:")
print(skewness)

# %%
# Split dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# %%
# Summary statistics for train data
print("Summary statistics for train data:")
print(train_data.describe())

# Summary statistics for test data
print("Summary statistics for test data:")
print(test_data.describe())

# %%
# Train Data Visualizations
# Parallel coordinate plot
train_data['State_numerical'] = train_data['State']
fig = px.parallel_coordinates(train_data, color='State_numerical', color_continuous_scale=px.colors.qualitative.Set1)
fig.update_layout(title="Parallel Coordinate Plot of Solar Dataset (Train Data)", xaxis_title="Features", yaxis_title="Values")
fig.show()

# Joint plot for train data
sns.jointplot(data=train_data, x='S1(Amp)', y='S2(Amp)', kind='scatter', hue='State')
plt.title("Joint Plot of S1(Amp) and S2(Amp) (Train Data)")
plt.show()

# Pair plot for train data
sns.pairplot(train_data, hue='State', palette='Dark2')
plt.suptitle("Pair Plot of Train Data", y=1.02)
plt.show()

# Correlation matrix heatmap for train data (Pearson)
correlation_matrix_train_pearson = train_data.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_train_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Pearson Correlation Matrix (Train Data)")
plt.show()

# Correlation matrix heatmap for train data (Spearman)
correlation_matrix_train_spearman = train_data.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_train_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Spearman Correlation Matrix (Train Data)")
plt.show()

# Correlation matrix heatmap for train data (Kendall)
correlation_matrix_train_kendall = train_data.corr(method='kendall')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_train_kendall, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Kendall Correlation Matrix (Train Data)")
plt.show()

# Distribution plot for train data
sns.histplot(data=train_data, x='S1(Amp)', kde=True)
plt.title("Distribution Plot of S1(Amp) (Train Data)")
plt.xlabel("S1(Amp)")
plt.ylabel("Frequency")
plt.show()

# %%
# Test Data Visualizations
# Parallel coordinate plot
test_data['State_numerical'] = test_data['State']
fig = px.parallel_coordinates(test_data, color='State_numerical', color_continuous_scale=px.colors.qualitative.Set1)
fig.update_layout(title="Parallel Coordinate Plot of Solar Dataset (Test Data)", xaxis_title="Features", yaxis_title="Values")
fig.show()

# Joint plot for test data
sns.jointplot(data=test_data, x='S1(Amp)', y='S2(Amp)', kind='scatter', hue='State')
plt.title("Joint Plot of S1(Amp) and S2(Amp) (Test Data)")
plt.show()

# Pair plot for test data
sns.pairplot(test_data, hue='State', palette='Dark2')
plt.suptitle("Pair Plot of Test Data", y=1.02)
plt.show()

# Correlation matrix heatmap for test data (Pearson)
correlation_matrix_test_pearson = test_data.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_test_pearson, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Pearson Correlation Matrix (Test Data)")
plt.show()

# Correlation matrix heatmap for test data (Spearman)
correlation_matrix_test_spearman = test_data.corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_test_spearman, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Spearman Correlation Matrix (Test Data)")
plt.show()

# Correlation matrix heatmap for test data (Kendall)
correlation_matrix_test_kendall = test_data.corr(method='kendall')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_test_kendall, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Kendall Correlation Matrix (Test Data)")
plt.show()

# Distribution plot for test data
sns.histplot(data=test_data, x='S1(Amp)', kde=True)
plt.title("Distribution Plot of S1(Amp) (Test Data)")
plt.xlabel("S1(Amp)")
plt.ylabel("Frequency")
plt.show()

# %%
# Point-biserial correlation
continuous_columns = ['S1(Amp)', 'S2(Amp)', 'S1(Volt)', 'S2(Volt)', 'Light(kiloLux)', 'Temp(degC)']
binary_column = 'Weather'
point_biserial_correlations = {column: pointbiserialr(dataset[column], dataset[binary_column])[0] for column in continuous_columns}
print("Point-Biserial Correlations:\n", point_biserial_correlations)

# Plot point-biserial correlations
plt.figure(figsize=(10, 6))
plt.bar(point_biserial_correlations.keys(), point_biserial_correlations.values(), color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Point-Biserial Correlation')
plt.title('Point-Biserial Correlation between Columns and Weather')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# Compute and visualize Shapiro-Wilk test for normality
for column in dataset.columns:
    stat, p = shapiro(dataset[column])
    print(f"Shapiro-Wilk test for {column}: p-value = {p}")

# Plot histograms for each column
for column in dataset.columns:
    sns.histplot(dataset[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()




