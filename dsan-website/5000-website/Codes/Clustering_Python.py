import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Framingham Heart Study data set
data = pd.read_csv("data/frmgham2.csv")
data.head()

# Check for missing data and calculate percentages
missing_data = data.isnull().mean()

# Drop columns with more than 50% missing data
threshold = 0.5
columns_to_drop = missing_data[missing_data > threshold].index
data.drop(columns=columns_to_drop, inplace=True)
data = data.iloc[:, 1:]

# Remove rows with missing data as their percentages are low
data.dropna(inplace=True)

# Set target variables
target = data.pop('DIABETES')

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Elbow method for K Means
inertia = []
K = range(1, 11)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(data_scaled)
    inertia.append(kmeanModel.inertia_)

# Plotting the elbow method
sns.lineplot(x=K, y=inertia, marker="o")
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# THIS WILL ITERATE OVER ONE HYPER-PARAMETER (GRID SEARCH)
# AND RETURN THE CLUSTER RESULT THAT OPTIMIZES THE SILHOUETTE SCORE
def maximize_silhouette(X, algo="birch", nmax=20, i_plot=False):

    # PARAM
    i_print = False

    # FORCE CONTIGUOUS
    X = np.ascontiguousarray(X)

    # LOOP OVER HYPER-PARAM
    params = []
    sil_scores = []
    sil_max = -10
    for param in range(2, nmax + 1):
        if algo == "birch":
            model = sklearn.cluster.Birch(n_clusters=param).fit(X)
            labels = model.predict(X)

        if algo == "ag":
            model = sklearn.cluster.AgglomerativeClustering(n_clusters=param).fit(X)
            labels = model.labels_

        if algo == "dbscan":
            param = 0.25 * (param - 1)
            model = sklearn.cluster.DBSCAN(eps=param).fit(X)
            labels = model.labels_

        if algo == "kmeans":
            model = sklearn.cluster.KMeans(n_clusters=param).fit(X)
            labels = model.predict(X)

        try:
            sil_scores.append(sklearn.metrics.silhouette_score(X, labels))
            params.append(param)
        except:
            continue

        if i_print:
            print(param, sil_scores[-1])

        if sil_scores[-1] > sil_max:
            opt_param = param
            sil_max = sil_scores[-1]
            opt_labels = labels

    print("OPTIMAL PARAMETER =", opt_param)

    if i_plot:
        fig, ax = plt.subplots()
        ax.plot(params, sil_scores, "-o")
        ax.set(xlabel='Hyper-parameter', ylabel='Silhouette')
        plt.show()

    return opt_labels

# Utility Plot function
def plot(X, color_vector):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=color_vector, alpha=0.5)  # , c=y
    ax.set(xlabel='Feature-1 (x_1)', ylabel='Feature-2 (x_2)',
           title='Cluster data')
    ax.grid()
    plt.show()

# Silhouette Method for K-Means
opt_labels = maximize_silhouette(data_scaled, algo="kmeans", nmax=15, i_plot=True)
plot(data_scaled, opt_labels)

# Confusion Matrix Heatmap
def plot_confusion_matrix(cm, class_names):
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig = plt.figure(figsize=(5, 4))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

# Final K-Means Clustering with Optimal k
kmeans_final = KMeans(n_clusters=2)
kmeans_clusters = kmeans_final.fit_predict(data_scaled)
class_names = ['Negative', 'Positive']

# Plot confusion matrix for K-Means
cm_kmeans = confusion_matrix(target, kmeans_clusters)
fig_kmeans = plot_confusion_matrix(cm_kmeans, class_names)
plt.title('Confusion Matrix for K-Means')
plt.show()

print(cm_kmeans)

precision = precision_score(target, kmeans_clusters)
recall = recall_score(target, kmeans_clusters)
f1 = f1_score(target, kmeans_clusters)
accuracy = accuracy_score(target, kmeans_clusters)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Silhouette Method for DBSCAN
opt_labels = maximize_silhouette(data_scaled, algo="dbscan", nmax=15, i_plot=True)
plot(data_scaled, opt_labels)

# Final Results for optimal eps of DBSCAN
optimal_eps = 3.5
dbscan_final = DBSCAN(eps=optimal_eps)
dbscan_clusters = dbscan_final.fit_predict(data_scaled)

cluster_positive_ratio = []
unique_clusters = set(dbscan_clusters) - {-1}  # Exclude noise if present

for cluster in unique_clusters:
    cluster_mask = dbscan_clusters == cluster
    positive_ratio = target[cluster_mask].mean()
    cluster_positive_ratio.append((cluster, positive_ratio))

optimal_cluster = sorted(cluster_positive_ratio, key=lambda x: x[1], reverse=True)[0][0]

binary_predictions = (dbscan_clusters == optimal_cluster).astype(int)

# Compute metrics
precision = precision_score(target, binary_predictions)
recall = recall_score(target, binary_predictions)
f1 = f1_score(target, binary_predictions)
accuracy = accuracy_score(target, binary_predictions)

# Confusion Matrix
cm = confusion_matrix(target, binary_predictions)
fig_kmeans = plot_confusion_matrix(cm, class_names)
plt.title('Confusion Matrix for DBSCAN')
plt.show()

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Hierarchical Clustering
Z = linkage(data_scaled, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()

# AGGLOMERATIVE CLUSTERING
opt_labels = maximize_silhouette(data_scaled, algo="ag", nmax=15, i_plot=True)
plot(data_scaled, opt_labels)

# Final Results for optimal clusters of Hierarchical clustering
optimal_clusters = 2
agglom_final = AgglomerativeClustering(n_clusters=optimal_clusters, affinity='euclidean', linkage='ward')
hierarchical_clusters = agglom_final.fit_predict(data_scaled)

# Plot confusion matrix for Hierarchical Clustering
cm_hierarchical = confusion_matrix(target, hierarchical_clusters)
fig_hierarchical = plot_confusion_matrix(cm_hierarchical, class_names)
plt.title('Confusion Matrix for Hierarchical Clustering')
plt.show()

precision = precision_score(target, hierarchical_clusters)
recall = recall_score(target, hierarchical_clusters)
f1 = f1_score(target, hierarchical_clusters)
accuracy = accuracy_score(target, hierarchical_clusters)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")