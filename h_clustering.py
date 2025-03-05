# hierarchical_clustering_project.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
import time

# 1. Introduction to Hierarchical Clustering
def generate_sample_data():
    """Generate sample data for clustering."""
    X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.60, random_state=0)
    return X

def plot_data(X):
    """Plot the sample data."""
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.title("Sample Data for Clustering")
    plt.show()

# 2. Implementing Agglomerative Clustering
def perform_clustering(X, method='single', n_clusters=3):
    """Perform hierarchical clustering using a specified linkage method."""
    Z = linkage(X, method)
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    return Z, clusters

def plot_clusters(X, clusters, title):
    """Plot the clusters."""
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
    plt.title(title)
    plt.show()

# 3. Exploring Linkage Criteria
def compare_linkage_methods(X):
    """Compare different linkage criteria (single, complete, average)."""
    methods = ['single', 'complete', 'average']
    fig, ax = plt.subplots(1, len(methods), figsize=(15, 5))
    
    for i, method in enumerate(methods):
        Z, clusters = perform_clustering(X, method)
        ax[i].scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50)
        ax[i].set_title(f"{method.capitalize()} Linkage Clustering")
    
    plt.show()

# 4. Ward's Criterion and Inversion
def perform_ward_clustering(X, n_clusters=3):
    """Perform hierarchical clustering using Ward's method."""
    Z = linkage(X, 'ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    return Z, clusters

def plot_dendrogram(Z, title):
    """Plot the dendrogram."""
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(title)
    plt.show()

# 5. Visualizing Dendrograms
def visualize_all_dendrograms(X):
    """Visualize dendrograms for all linkage criteria."""
    methods = ['single', 'complete', 'average', 'ward']
    
    for method in methods:
        Z = linkage(X, method)
        plot_dendrogram(Z, f"Dendrogram for {method.capitalize()} Linkage")

# 6. Advanced Topics and Applications
def discuss_advanced_topics(X):
    """Discuss advanced topics like inversion, practical applications, and limitations with examples."""
    print("\nAdvanced Topics and Applications:")
    
    # 1. Counterexample for Inversion
    print("\n1. Counterexample for Inversion:")
    print("- Inversion occurs when the height function in the dendrogram is not monotonous.")
    print("- Example: Consider a dataset where clusters merge in a non-monotonic way due to overlapping distributions.")
    print("- Let's simulate a dataset where inversion occurs:")

    # Simulate a dataset where inversion might occur
    X_inversion = np.array([[1, 2], [1.5, 2], [5, 6], [5.5, 6], [10, 11], [10.5, 11]])
    Z_inversion = linkage(X_inversion, 'ward')
    plot_dendrogram(Z_inversion, "Dendrogram Showing Inversion")
    print("- Observe the dendrogram: The height function may not increase monotonically due to overlapping clusters.")
    print("- This can lead to misinterpretation of the hierarchical structure.")

    # 2. Practical Applications with Example
    print("\n2. Practical Applications with Example:")
    print("- Hierarchical clustering is widely used in biology for phylogenetic tree construction.")
    print("- Example: Clustering gene expression data to identify groups of co-expressed genes.")
    print("- Let's simulate gene expression data and apply hierarchical clustering:")

    # Simulate gene expression data
    np.random.seed(42)
    gene_data = np.random.rand(10, 5)  # 10 genes, 5 samples
    Z_gene = linkage(gene_data, 'ward')
    plot_dendrogram(Z_gene, "Dendrogram for Gene Expression Data")
    print("- The dendrogram helps biologists identify groups of genes with similar expression patterns.")
    print("- This can lead to insights into gene regulatory networks.")

    # 3. Limitations with Concrete Scenarios
    print("\n3. Limitations with Concrete Scenarios:")
    print("- Hierarchical clustering is computationally expensive for large datasets.")
    print("- Example: Clustering 1 million data points with Ward's method can take hours or days.")
    print("- Let's simulate a large dataset and measure the time taken for clustering:")

    X_large, _ = make_blobs(n_samples=10000, centers=5, cluster_std=1.0, random_state=42)
    print(f"- Dataset size: {X_large.shape[0]} points")
    start_time = time.time()
    Z_large = linkage(X_large, 'ward')
    end_time = time.time()
    print(f"- Time taken for clustering: {end_time - start_time:.2f} seconds")

    print("\n- Sensitivity to noise and outliers:")
    print("- Example: Adding outliers to a dataset can drastically change the clustering structure.")
    print("- Let's add an outlier to the sample data and observe the effect:")

    X_outlier = np.vstack([X, [[20, 20]]])  # Add an outlier
    Z_outlier = linkage(X_outlier, 'ward')
    plot_dendrogram(Z_outlier, "Dendrogram with Outlier")
    print("- The outlier creates a separate branch in the dendrogram, disrupting the natural cluster structure.")

    

# Main function to run the project
def main():
    # Generate sample data
    X = generate_sample_data()
    
    # Plot the sample data
    plot_data(X)
    
    # Perform clustering using single linkage
    Z_single, clusters_single = perform_clustering(X, 'single')
    plot_clusters(X, clusters_single, "Single Linkage Clustering")
    
    # Compare different linkage criteria
    compare_linkage_methods(X)
    
    # Perform clustering using Ward's method
    Z_ward, clusters_ward = perform_ward_clustering(X)
    plot_clusters(X, clusters_ward, "Ward's Method Clustering")
    
    # Plot dendrogram for Ward's method
    plot_dendrogram(Z_ward, "Dendrogram for Ward's Method")
    
    # Visualize dendrograms for all linkage criteria
    visualize_all_dendrograms(X)
    
    # Discuss advanced topics
    discuss_advanced_topics(X)

if __name__ == "__main__":
    main()
