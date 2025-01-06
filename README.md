# Chronic Disease Analysis and Prediction Using Machine Learning and NLP

## Overview
This project focuses on analyzing and predicting two critical chronic conditions—**Type 2 Diabetes (T2D)** and **Cardiovascular Diseases (CVD)**—using structured datasets and natural language processing (NLP) techniques. By integrating advanced machine learning models, clustering methods, and sentiment analysis, this study aims to uncover risk factors, classify disease outcomes, and provide actionable insights to improve disease management and prevention.

The website for this project can be accessed here: [Chronic Disease Study Website](https://michaelx1010.github.io/Chronic-Disease-Study/).

---

## Objectives
The key objectives of this project are:
1. To identify key lifestyle, dietary, genetic, and environmental factors linked to T2D and CVD.
2. To evaluate the effectiveness of machine learning models, such as Naive Bayes, Decision Trees, and Random Forests, for predicting chronic disease outcomes.
3. To explore clustering techniques (e.g., K-Means, DBSCAN, and Hierarchical Clustering) for patient segmentation.
4. To analyze community discussions from the **r/diabetes** subreddit using NLP techniques for insights on medications, lifestyle changes, and patient sentiment.
5. To compare dimensionality reduction methods like PCA and t-SNE for improving data interpretability.
6. To effectively communicate findings using interactive and static data visualizations.

---

## Datasets
The analysis uses a combination of structured and unstructured data:
- **Structured Datasets**:
  - [Chronic Disease Indicators (CDI)](https://www.cdc.gov/cdi/)
  - [Framingham Heart Study](https://www.framinghamheartstudy.org/)
  - [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

- **Unstructured Dataset**:
  - Text data from the **r/diabetes** subreddit, covering submissions and comments from June 2023 to July 2024.

---

## Methods and Tools

### Machine Learning Models
- **Naive Bayes Classifier**: For probabilistic classification of cardiovascular disease risks.
- **Decision Trees and Random Forests**: For classification and regression analysis of glucose levels and disease risks.
- **Clustering Models**:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering

### Natural Language Processing (NLP)
- **TF-IDF Analysis**: To extract key themes and topics from subreddit discussions.
- **Sentiment Analysis**: To evaluate community sentiment around lifestyle changes, medications, and research developments.

### Dimensionality Reduction
- **PCA (Principal Component Analysis)** and **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: For visualization of high-dimensional data and pattern discovery.

### Data Visualization
- Tools: **Matplotlib**, **Seaborn**, **Plotly**, and **Tableau** for creating static and interactive visualizations.

---

## Key Findings
- **Naive Bayes Classifier**: Achieved an AUC of 0.97, showcasing its strong potential for cardiovascular disease classification.
- **Clustering**: K-Means performed moderately well, while DBSCAN and Hierarchical Clustering showed lower precision in distinguishing patient groups.
- **Decision Trees and Random Forests**: Highlighted variability in performance, emphasizing the need for fine-tuning to balance complexity and generalizability.
- **NLP Analysis**:
  - Positive sentiment was predominantly associated with lifestyle changes (e.g., diet and exercise).
  - Mixed sentiment surrounded medications, especially GLP-1 agonists, driven by concerns over side effects and costs.
  - A spike in positive sentiment in December 2023 coincided with groundbreaking research in insulin production.

---

## Future Work
1. Integrate ensemble methods, such as boosting, to enhance classification accuracy.
2. Optimize clustering algorithms for better scalability and precision in identifying patient subgroups.
3. Expand NLP analysis with transformer-based models like BERT and GPT for advanced text understanding.
4. Incorporate additional external data sources, such as clinical trials or prescription statistics, for richer analysis.
5. Explore ethical and interpretability considerations in machine learning for healthcare.

---

## How to Run the Project

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- Required Python libraries:
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - NLTK
  - PyTorch
  - Plotly
