# %%
import pandas as pd
import scipy
import numpy as np


# %%
from scipy.stats import skew

# %%
dataset = pd.read_csv("/content/Solar_categorical.csv")

# %%
print (type(dataset))

# %%
numeric_cols = dataset.select_dtypes(include=np.number).columns
skewness = dataset[numeric_cols].apply(lambda x: x.skew())

print("Skewness of each numeric column:")
print(skewness)

# %%
!pip install plotly


# %%
import plotly.express as px

# %%
import pandas as pd
import plotly.express as px

# Load the dataset
dataset = pd.read_csv("/content/Solar_categorical.csv")

# Map categorical values to numerical values
state_mapping = {'Normal': 0, 'Line-line': 1, 'Open': 2}
dataset['State_numerical'] = dataset['State'].map(state_mapping)

# Create parallel coordinate plot
fig = px.parallel_coordinates(dataset, color='State_numerical', color_continuous_scale=px.colors.qualitative.Set1)

# Customize the plot layout
fig.update_layout(
    title="Parallel Coordinate Plot of Solar Dataset",
    xaxis_title="Features",
    yaxis_title="Values",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

# Show the plot
fig.show()


# %%
print(dataset.columns)


# %% [markdown]
# # New Section

# %% [markdown]
# # New Section

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv")

# Specify the variables you want to visualize
# For example, let's visualize the relationship between 'S1(Amp)' and 'S2(Amp)'
x = ' S1(Amp)'
y = 'S2(Amp)'

# Create joint plot
sns.jointplot(data=dataset, x=x, y=y, kind='scatter', hue='State')

# Show the plot
plt.show()


# %%


# Calculate the correlation matrix
correlation_matrix = dataset.corr()

# Display the correlation matrix
print(correlation_matrix)


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate the correlation matrix
correlation_matrix = dataset.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt



# Specify the variable you want to plot
variable_of_interest = ' S1(Amp)'  # Replace 'S1(Amp)' with the variable you want to plot

# Create the distribution plot
plt.figure(figsize=(8, 6))
sns.histplot(data=dataset, x=variable_of_interest, kde=True)  # kde=True adds a kernel density estimate
plt.title("Distribution Plot of {}".format(variable_of_interest))
plt.xlabel(variable_of_interest)
plt.ylabel("Frequency")
plt.show()


# %%
import pandas as pd



# Check for missing values
missing_values = dataset.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values
# For example, let's impute missing values with the mean of each column
dataset_imputed = dataset.fillna(dataset.mean())

# Alternatively, you can drop rows with missing values using dropna()
# dataset_clean = dataset.dropna()

# Verify that missing values have been handled
print("\nAfter handling missing values:\n", dataset_imputed.head())


# %%
# Load dataset
dataset = pd.read_csv("/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv")
print(dataset.columns)

# %%
# Example feature engineering
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load dataset
dataset = pd.read_csv("/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv")
# Remove leading spaces from column names
dataset.columns = dataset.columns.str.strip()
# Define features and target variable
X = dataset.drop(columns=['State'])  # Features (all columns except the 'State' column)
y = dataset['State']  # Target variable ('State' column)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['S1(Amp)', 'S2(Amp)', 'S1(Volt)', 'S2(Volt)', 'Light(kiloLux)', 'Temp(degC)']
categorical_features = ['Weather']




numeric_transformer = StandardScaler()
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])





# %%
# Combine the transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# %%

# Apply feature engineering
X_processed = preprocessor.fit_transform(X)

# %%
# 1. Print the transformed feature matrix
print(X_processed)

# 2. Summary statistics
print("Summary statistics:")
print(pd.DataFrame(X_processed).describe())



# %%
# 3. Visualization
import seaborn as sns
import matplotlib.pyplot as plt



# %%
print(X_processed.shape)


# %%
# Plot histograms of transformed numeric features
numeric_features_transformed = pd.read_csv(X_processed, columns=numeric_features)
numeric_features_transformed.hist(figsize=(12, 8))
plt.suptitle("Histograms of Transformed Numeric Features")
plt.show()


# %%
# 4. Check for missing values
missing_values = pd.DataFrame(X_processed).isnull().sum()
if missing_values.any():
    print("There are missing values in the transformed features.")
else:
    print("There are no missing values in the transformed features.")

# %%

#import pandas
import pandas as pd

# read csv file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# replacing values
df['Weather'].replace(['Sunny', 'cloudy'],
                        [0, 1], inplace=True)

# %%

#import pandas
import pandas as pd

# read csv file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# replacing values
df['State'].replace(['Line-line', 'Open', 'Normal'],
                        [0, 1,2], inplace=True)

# %%


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()
#pearson correlation


# %%
# Compute Spearman correlation matrix
spearman_correlation = df.corr(method='spearman')

# Visualize Spearman correlation matrix
sns.heatmap(spearman_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation')
plt.show()


# %%
# Compute Kendall correlation matrix
kendall_correlation = df.corr(method='kendall')

# Visualize Kendall correlation matrix
sns.heatmap(kendall_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Kendall Correlation')
plt.show()


# %%
!pip install pingouin

# %%
# Use pairplot for a quick correlogram
sns.pairplot(df)
plt.show()


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'dataset' is your DataFrame
df = pd.read_csv('/df_copy (2).csv')
# Compute Pearson correlation matrix
pearson_correlation = df.corr()

# Visualize Pearson correlation matrix with a different color palette
sns.heatmap(pearson_correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation')
plt.show()


# Compute Spearman correlation matrix
spearman_correlation = df.corr(method='spearman')

# Visualize Spearman correlation matrix with a different color palette
sns.heatmap(spearman_correlation, annot=True, cmap='YlGnBu', fmt=".2f")
plt.title('Spearman Correlation')
plt.show()


# Compute Kendall correlation matrix
kendall_correlation = df.corr(method='kendall')

# Visualize Kendall correlation matrix with a different color palette
sns.heatmap(kendall_correlation, annot=True, cmap='magma', fmt=".2f")
plt.title('Kendall Correlation')
plt.show()


# Since partial correlation does not inherently produce a visualization, you may use a different color palette in the other plots.

# Since point-biserial correlation does not inherently produce a visualization, you may use a different color palette in the other plots.

# Since Cramér's V does not inherently produce a visualization, you may use a different color palette in the other plots.

# Since distance correlation does not inherently produce a visualization, you may use a different color palette in the other plots.


# Use pairplot for a quick correlogram
sns.pairplot(df, palette='Dark2')
plt.show()


# %%
import pandas as pd

# Assuming 'dataset' is your DataFrame

# Get data types of each column
data_types = df.dtypes

# Filter columns with numeric data types (excluding 'object' type, which typically represents categorical variables)
numeric_columns = data_types[data_types != 'object'].index

# Check the unique value counts for each numeric column to determine if they are continuous
for column in numeric_columns:
    unique_values_count = df[column].nunique()
    print(f"Column '{column}' has {unique_values_count} unique values")

# Optionally, you can set a threshold to differentiate continuous variables from discrete ones
# For example, if a column has more than 10 unique values, you may consider it continuous
continuous_variables = [column for column in numeric_columns if df[column].nunique() > 10]

print("Continuous Variables:", continuous_variables)


# %%
# Assuming 'df' is your DataFrame
df.to_csv('df_copy.csv', index=False)


# %%
# Assuming 'df' is your DataFrame
num_rows = df.shape[0]
print("Number of rows in the DataFrame:", num_rows)


# %%

#import pandas
import pandas as pd

# read csv file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# replacing values
df['State'].replace(['Normal', 'Open', 'Line-line'],
                        [0,1,2], inplace=True)

# %%

#import pandas
import pandas as pd

# read csv file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# replacing values
df['Weather'].replace(['Sunny', 'Cloudy'],
                        [0, 1], inplace=True)

# %%
# Assuming 'df' is your DataFrame
num_rows = df.shape[0]
print("Number of rows in the DataFrame:", num_rows)

# %%
# Assuming 'df' is your DataFrame
df.to_csv('df_copy.csv', index=False)


# %%
import pandas as pd

# read csv file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# Print the unique values in the 'State' column to verify its contents
print("Unique values in 'State' column before replacement:", df['State'].unique())
print("Unique values in 'Weather' column before replacement:", df['Weather'].unique())
# Replacing values
df['State'].replace(['Normal', 'Open', 'Line-line'], [0, 1, 2], inplace=True)
# replacing values
df['Weather'].replace(['Sunny', 'Cloudy'],
                        [0, 1], inplace=True)

# Print the unique values in the 'State' column after replacement
print("Unique values in 'State' column after replacement:", df['State'].unique())
print("Unique values in 'Weather' column after replacement:", df['Weather'].unique())

# %%
# Assuming 'df' is your DataFrame
df.to_csv('df.csv', index=False)
#/content/drive/MyDrive/PV_fault_Python-master/df.csv

# %%
from scipy import stats
import pandas as pd

# Assuming 'df' is your DataFrame

# Convert 'Weather' column to numeric representation
df['Weather'] = pd.to_numeric(df['Weather'], errors='coerce')

# Drop rows containing NaN values
df.dropna(inplace=True)

# Selecting the continuous columns and the binary column
continuous_columns = [0, 1, 2, 3, 4, 5, 7]  # Adjust column indices based on 0-based indexing
binary_column = 'Weather'

# Calculate point-biserial correlation for each continuous variable
point_biserial_correlations = {}
for column_index in continuous_columns:
    # Extract the continuous variable and the binary variable
    continuous_variable = df.iloc[:, column_index]
    binary_variable = df[binary_column]

    # Check if both variables have at least two distinct values
    if len(continuous_variable.unique()) > 1 and len(binary_variable.unique()) > 1:
        # Calculate point-biserial correlation
        correlation, p_value = stats.pointbiserialr(continuous_variable, binary_variable)
        point_biserial_correlations[column_index] = correlation
    else:
        print(f"Skipping column {column_index} due to insufficient unique values.")

# Print point-biserial correlations
for column_index, correlation in point_biserial_correlations.items():
    print(f"Point-Biserial Correlation between column {column_index} and 'Weather': {correlation}")

# %%
import pandas as pd

# Point-biserial correlation values
correlation_values = {
    ' S1(Amp)': -0.26010945509477423,
    'S2(Amp)': -0.3506584251590675,
    'S1(Volt)': 0.17367066693762917,
    'S2(Volt)': -0.2187876571403985,
    'Light(kiloLux)': -0.5144058973521928,
    'Temp(degC)': -0.20030357796221368,
    'State': 0.0016329989680843292
}

# Create DataFrame from dictionary
correlation_df = pd.DataFrame(correlation_values, index=['Point-Biserial Correlation'])

# Transpose DataFrame to swap rows and columns
correlation_df = correlation_df.transpose()

# Rename index column
correlation_df.index.name = 'Column'

# Print the resulting DataFrame
print(correlation_df)


# %%
import matplotlib.pyplot as plt

# Point-biserial correlation values
correlation_values = {
    ' S1(Amp)': -0.26010945509477423,
    'S2(Amp)': -0.3506584251590675,
    'S1(Volt)': 0.17367066693762917,
    'S2(Volt)': -0.2187876571403985,
    'Light(kiloLux)': -0.5144058973521928,
    'Temp(degC)': -0.20030357796221368,
    'State': 0.0016329989680843292
}

# Create DataFrame from dictionary
correlation_df = pd.DataFrame(correlation_values, index=['Point-Biserial Correlation'])

# Transpose DataFrame to swap rows and columns
correlation_df = correlation_df.transpose()

# Plot
plt.figure(figsize=(10, 6))
plt.bar(correlation_df.index, correlation_df['Point-Biserial Correlation'], color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Point-Biserial Correlation')
plt.title('Point-Biserial Correlation between Columns and Weather')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt

# Column names
columns = ['S1(Amp)', 'S2(Amp)', 'S1(Volt)', 'S2(Volt)', 'Light(kiloLux)', 'Temp(degC)', 'State']

# Point-biserial correlation values
correlation_values = [-0.26010945509477423, -0.3506584251590675, 0.17367066693762917,
                      -0.2187876571403985, -0.5144058973521928, -0.20030357796221368,
                      0.0016329989680843292]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(columns, correlation_values, color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Point-Biserial Correlation')
plt.title('Point-Biserial Correlation between Columns and Weather')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# %%
from scipy.stats import shapiro

for column in df.columns:
    stat, p = shapiro(df[column])
    print(f"Shapiro-Wilk test for {column}: p-value = {p}")


# %%
import seaborn as sns

# Plot histograms for each column
for column in df.columns:
    sns.histplot(df[column], kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()


# %%
!pip install pingouin


# %%
!pip install researchpy


# %%
import researchpy as rp
import pandas as pd

# Assuming 'dataset' is a contingency table
dataset = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/Solar_categorical.csv')

# Compute Cramer's V
cramers_v = rp.crosstab(dataset['Weather'], dataset['State'], prop= 'cell')

# Print Cramer's V
print("Cramer's V:", cramers_v)


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Provided contingency table
contingency_table = pd.DataFrame({
    'State': ['Line-line', 'Normal', 'Open', 'All'],
    'Cloudy': [16.73, 16.67, 16.73, 50.13],
    'Sunny': [16.60, 16.67, 16.60, 49.87],
    'All': [33.33, 33.33, 33.33, 100.00]
})

# Set 'State' column as index
contingency_table.set_index('State', inplace=True)

# Drop the 'All' row and column (optional, if you don't want to include in heatmap)
contingency_table.drop('All', axis=0, inplace=True)
contingency_table.drop('All', axis=1, inplace=True)

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, cmap='viridis', fmt='.2f', cbar=True)
plt.title("Cramer's V Heatmap for Weather vs State")
plt.xlabel('Weather')
plt.ylabel('State')
plt.show()


# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd

# Assuming 'df' is your DataFrame
# Replace '/path/to/your/data.csv' with the actual path to your CSV file
df = pd.read_csv('/content/drive/MyDrive/PV_fault_Python-master/df.csv')

# Display summary statistics
summary_stats = df.describe()
print("Summary statistics:")
print(summary_stats)



