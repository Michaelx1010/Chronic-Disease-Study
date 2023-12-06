#Loading required packages
library(RANN)
library(caret)
library(mice)
library(tidyverse)
library(ggplot2)
library(plotly)
library(ggthemes)
library(DT)
library(ggfortify)
library(Rtsne)
library(cluster)
library(factoextra)
library(fpc)
library(dbscan)

#Load the framingham heart study data set
data <- read_csv("data/frmgham2.csv")
head(data)
glimpse(data)

# Check for missing data
sum(is.na(data))

data %>% 
  summarise_all(~sum(is.na(.)))

# Check for missing data percentages
data %>% 
  summarise_all(~sum(is.na(.))/nrow(data))

# Drop data columns that have over 50% of missing data
threshold <- 0.5

# Calculate the percentage of missing values for each column
missing_percentages <- colMeans(is.na(data))

# Identify columns that exceed the threshold
columns_to_drop <- names(data)[missing_percentages > threshold]

# Drop the identified columns from the dataframe
data <- data %>% select(-(columns_to_drop))

data <- na.omit(data)

target <- data$DIABETES

# Remove the 'DIABETES' column
data <- data %>% select(-DIABETES)

# Function to check if a column is binary
is_binary <- function(column) {
  unique_values <- unique(column)
  length(unique_values) == 2 && all(unique_values %in% c(0, 1))
}

# Identify binary columns
binary_columns <- sapply(data, is_binary)

# Remove binary columns from the data frame
data <- data[, !binary_columns]

# Remove the first two columns

data <- data %>%
  select(-RANDID, -SEX)

glimpse(data)



### K Means Clustering

# Elbow Method
# Normalize data
data_n <- scale(data)

inertia <- numeric(10)
for(k in 1:10){
  model <- kmeans(data_n, centers = k)
  inertia[k] <- model$tot.withinss
}

# Create a data frame for plotting
elbow_df <- data.frame(k = 1:10, Inertia = inertia)

# Plot using ggplot
ggplot(elbow_df, aes(x = k, y = Inertia)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Elbow Method for Optimal K", x = "Number of Clusters", y = "Inertia")


# Silhouette Method

sil_width <- numeric(10)
for(k in 2:10){
  model <- kmeans(data_n, centers = k)
  sil_width[k] <- mean(silhouette(model$cluster, dist(data))[, 3])
}

# Create a data frame for plotting
silhouette_df <- data.frame(k = 2:10, SilhouetteWidth = sil_width[-1])

# Plot using ggplot
ggplot(silhouette_df, aes(x = k, y = SilhouetteWidth)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Silhouette Method for Optimal K", x = "Number of Clusters", y = "Silhouette")

# Final Results for optimal K of K-Means

k <- 2

data_k <- data

final_model <- kmeans(data_n, centers = k)

data_k$cluster <- final_model$cluster
data_k$cluster <- ifelse(data_k$cluster == 1, 0, 1)
data_k$Target <- target
table <- table(Cluster = data_k$cluster, Target = target)
table

purity <- sum(apply(table, 1, max)) / nrow(data_k)
print(paste("Purity is: ", purity))

# Perform PCA on the dataset for results visualization
pca_res <- prcomp(data_k[, -which(names(data_k) %in% c("cluster", "Target"))], scale. = TRUE)
data_pca <- as.data.frame(pca_res$x)
data_pca$cluster <- as.factor(data_k$cluster)

# Plot the first two principal components with ggplot2
p <- ggplot(data_pca, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.5) + 
  theme_minimal() +
  labs(title = "Cluster Visualization on PCA-reduced Data", color = "Cluster") +
  xlab("PC1") +
  ylab("PC2") +
  scale_color_discrete(name = "Cluster")  
p

## DBSCAN

# Normalize data
data_n <- scale(data)

eps_values <- seq(0.1, 2, by = 0.1) 
sil_scores <- c()

for (eps in eps_values) {
  dbscan_res <- dbscan(data_n, eps = eps, minPts = 4)
  if (max(dbscan_res$cluster) > 1) { 
    sil_score <- silhouette(dbscan_res$cluster, dist(data))
    sil_scores <- c(sil_scores, mean(sil_score[, "sil_width"]))
  } else {
    sil_scores <- c(sil_scores, NA) 
  }
}

optimal_eps <- eps_values[which.max(sil_scores)]
optimal_eps

plot_data <- data.frame(eps = eps_values, silhouette = sil_scores)

ggplot(plot_data, aes(x = eps, y = silhouette)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Silhouette method for Different eps Values", x = "eps", y = "Silhouette")

data_d <- data

optimal_eps <- 0.7
dbscan_result <- dbscan(data_n, eps = optimal_eps, minPts = 4)

data_d$target <- target
data_d$cluster <- dbscan_result$cluster
comparison_table <- table(Cluster = data_d$cluster, Target = data_d$target)
print(comparison_table)

purity <- sum(apply(comparison_table, 1, max)) / nrow(data_d)
print(paste("Purity of clusters:", purity))


### Hierarchical clustering

# Perform hierarchical clustering 
h <- hclust(dist(data_n), method = "ward.D2")

# Draw the dendrogram
plot(h, main = "Hierarchical Clustering", sub = "", xlab = "")

sil <- sapply(2:10, function(k) {
  c <- cutree(h, k)
  mean(silhouette(c, dist(data))[, "sil_width"])
})

oc <- which.max(sil)

# Plot the silhouette scores for different numbers of clusters
plot(2:10, sil, type = 'b', xlab = "Number of Clusters", ylab = "Silhouette",
     main = "Silhouette Scores for Different Numbers of Clusters")

clusters <- cutree(h, 2)

table <- table(Cluster = clusters, Target = target)

print(table)

purity <- sum(apply(table, 1, max)) / sum(table)
print(paste("Purity of clusters:", purity))




