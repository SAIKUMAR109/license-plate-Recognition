import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the newly uploaded dataset
try:
    # Load dataset
    data = pd.read_csv('/Licplatesrecognition_train.csv')

    # Explore the dataset
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Information:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())

    # Step 2: Data Pre-processing
    # Handle missing values
    data.fillna(method='ffill', inplace=True)

    # Encode categorical columns (if any)
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])

    # Normalize numerical features
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = MinMaxScaler().fit_transform(data[numerical_columns])

    # Step 3: Exploratory Data Analysis (EDA)
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Adjust this column name based on dataset structure for histogram
    example_column = numerical_columns[0] if len(numerical_columns) > 0 else None
    if example_column:
        plt.figure(figsize=(8, 6))
        data[example_column].hist(bins=30)
        plt.title(f"Distribution of {example_column}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No numerical column found for histogram plotting.")

except Exception as e:
    print(f"An error occurred while processing the dataset: {e}")
