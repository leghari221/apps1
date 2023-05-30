import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

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

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_data, x='species', y=selected_feature, palette='Set1')
st.pyplot()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=iris_data, x='species', y=selected_feature, palette='Set1')
st.pyplot()

# Clustering
clustering_enabled = st.checkbox('Enable Clustering')

if clustering_enabled:
    # Select the number of clusters
    num_clusters = st.slider('Select the number of clusters', min_value=2, max_value=5, value=3)

    # Perform K-means clustering
    X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)

    # Add cluster labels to the dataset
    iris_data['cluster'] = kmeans.labels_

    # Scatter plot with cluster coloring
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=iris_data, x=selected_feature, y='species', hue='cluster', palette='Set1')
    st.pyplot()

    # Display cluster statistics
    st.subheader('Cluster Statistics')
    cluster_stats = iris_data.groupby('cluster')[selected_feature].describe()
    st.dataframe(cluster_stats)

# Display the filtered data
st.subheader('Filtered Data')
st.dataframe(iris_data)

