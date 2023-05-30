import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris_data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Set the page title
st.title('Iris Dataset Visualization')

# Display the dataset
st.subheader('Iris Dataset')
st.dataframe(iris_data)

# Interactive visualization
st.subheader('Interactive Visualization')

# Select the feature to visualize
feature_options = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
selected_feature = st.selectbox('Select a feature', feature_options)

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_data, x=selected_feature, y='species', hue='species', palette='Set1')
st.pyplot()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=iris_data, x=selected_feature, hue='species', element='step', kde=True)
st.pyplot()

# Bar plot
species_counts = iris_data['species'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=species_counts.index, y=species_counts.values, palette='Set1')
st.pyplot()
