import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

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

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data.head()

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)

# Plot the PCA results
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])

plt.title('PCA Results')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.figure()
sns.barplot(x=[f"PC{i+1}" for i in range(pca.n_components_)], y=pca.explained_variance_ratio_)
plt.title('Explained Variance by PCA Components')
plt.ylabel('Explained Variance Ratio')

plt.show()

# Create a DataFrame with the explained variance and cumulative variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

pca_table = pd.DataFrame({'Principal Component': [f"PC{i+1}" for i in range(len(explained_variance))],
                          'Explained Variance': explained_variance,
                          'Cumulative Variance': cumulative_variance})

print(pca_table)

# Get the loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a new matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the loadings for each feature as arrows
for i, (loading1, loading2) in enumerate(loadings):
    ax.arrow(0, 0, loading1, loading2, head_width=0.05, head_length=0.1, length_includes_head=True, color='red')
    plt.text(loading1 * 1.2, loading2 * 1.2, data.columns[i], color='black', ha='center', va='center')

# Set plot labels and title
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_title('PCA Biplot')
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)
ax.grid(True)

# Show the plot
plt.show()

# t-SNE analysis with different perplexity values
ps = [5, 30, 50, 100]

for p in ps:
    tsne = TSNE(n_components=2, perplexity=p, random_state=42)
    tsne_result = tsne.fit_transform(data_scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])

    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2')
    plt.title(f't-SNE with Perplexity {p}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()