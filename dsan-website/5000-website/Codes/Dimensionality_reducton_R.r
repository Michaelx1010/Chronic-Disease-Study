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
```

## Introduction



## Data preparation

#Load the framingham heart study data set
data <- read_csv("data/frmgham2.csv")
head(data)
glimpse(data)



### Missing data


# Check for missing data
sum(is.na(data))



data %>% 
  summarise_all(~sum(is.na(.)))



# Check for missing data percentages
data %>% 
  summarise_all(~sum(is.na(.))/nrow(data))

# Drop data columns that has over 50% of missing data
threshold <- 0.5

# Calculate the percentage of missing values for each column
missing_percentages <- colMeans(is.na(data))

# Identify columns that exceed the threshold
columns_to_drop <- names(data)[missing_percentages > threshold]

# Drop the identified columns from the dataframe
data <- data %>% select(-(columns_to_drop))

head(data)

# Check for missing data percentages
data %>% 
  summarise_all(~sum(is.na(.))/nrow(data))

# Remove the missing data directly since the percentagies of missing values are low
data <- na.omit(data)

sum(is.na(data))


## PCA method

# Scale the data
data_scaled <- scale(data)

# Apply PCA
pca_result <- prcomp(data_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA results
summary(pca_result)

# Plot the PCA results
plot(pca_result)
abline(v = 1:37, col = "lightgray", lty = 2)
axis(1, at = 1:37, labels = TRUE)


### PCA Biplot visualization

loadings <- data.frame(pca_result$rotation)

p <- ggplot() +
  geom_segment(data = loadings, aes(x = 0, y = 0, xend = PC1, yend = PC2), arrow = arrow(length = unit(0.02, "npc")), color = 'red') +
  geom_text(data = loadings, aes(x = PC1, y = PC2, label = rownames(loadings)), hjust = 1.2, vjust = 1.2) +
  theme_minimal() +
  labs(x = "First Principal Component", y = "Second Principal Component", title = "PCA Loadings Biplot") +
  coord_equal() 

p + theme(
  plot.title = element_text(size = 20),  
  axis.text = element_text(size = 14),  
  axis.title = element_text(size = 16)
)


## t-SNE method


# Run t-SNE with a range of perplexity values
perplexities <- c(5, 30, 50, 100)
for (perplexity in perplexities) {
  set.seed(42)
  tsne_results <- Rtsne(data_scaled, dims = 2, perplexity = perplexity, verbose = TRUE)
  
# Create a data frame for plotting
  tsne_data <- data.frame(tsne_results$Y)
  colnames(tsne_data) <- c("TSNE1", "TSNE2")
  
# Plot the t-SNE outputs
  p <- ggplot(tsne_data, aes(x = TSNE1, y = TSNE2)) +
    geom_point() +
    ggtitle(paste("t-SNE with Perplexity", perplexity))
  
  print(p)
}



### PCA/t-SNE Comparison


pca_data <- data.frame(pca_result$x)
ggplot(pca_data, aes(x = PC1, y = PC2)) +
  geom_point() +
  ggtitle("PCA Results")