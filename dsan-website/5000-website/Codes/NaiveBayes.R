# Load required libraries and implement the cleaned data
library(tidyverse)
library(ggplot2)
library(plotly)
library(ggthemes)
library(DT)
library(e1071)
library(caTools)
library(caret)
library(mlbench)
library(cvms)

#us_chronic <- read_csv("data/us_chronic.csv")

data(PimaIndiansDiabetes)
data <- PimaIndiansDiabetes

# Setting the ratio
fractionTraining   <- 0.60
fractionValidation <- 0.20
fractionTest       <- 0.20

# Compute sample sizes.
sampleSizeTraining   <- floor(fractionTraining   * nrow(data))
sampleSizeValidation <- floor(fractionValidation * nrow(data))
sampleSizeTest       <- floor(fractionTest       * nrow(data))

# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(nrow(data)), size = sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(data)), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size = sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

# Finally, output the three dataframes for training, validation, and test.
data_training   <- data[indicesTraining, ]
data_validation <- data[indicesValidation, ]
data_test       <- data[indicesTest, ]

# calculate correlation matrix
correlationMatrix <- cor(data_training[,1:8])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=data_training, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(data_training[,1:8], data_training[,9], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))


nb <- data_training %>%
  select(diabetes, glucose, age, mass, pregnant, insulin, pedigree, triceps, pressure)
glimpse(nb)


# Fitting Naive Bayes Model 
# to training dataset
set.seed(120)  # Setting Seed
classifier_cl <- naiveBayes(diabetes ~ ., data = data_training)

# Predicting on test data'
pred <- predict(classifier_cl, newdata = data_test)

# Confusion Matrix
cm <- table(data_test$diabetes, pred)

# Model Evaluation
model <- confusionMatrix(cm)
model

# Calculate F1
f1_score <- 2 * (0.8901 * 0.7431) / (0.8901 + 0.7431)
f1_score

# Modified this code with the aid of gpt, it is a great way to visualize accuracies
# Creating a data frame with actual and predicted values
result_df <- data.frame(Actual = data_test$diabetes, Predicted = pred)

# Creating a variable to identify correct and incorrect predictions
result_df$Correct <- ifelse(result_df$Actual == result_df$Predicted, "Correct", "Incorrect")

# Counting occurrences for each combination of actual and predicted values
counts <- table(result_df$Correct)

# Creating a data frame for plotting
plot_data <- data.frame(Category = names(counts), Count = as.numeric(counts))

# Plotting
ggplot(plot_data, aes(x = Category, y = Count, group = 1)) +
  geom_line(aes(color = Category), size = 1.5) +
  geom_point(aes(color = Category), size = 3) +
  labs(title = "Actual vs. Predicted",
       x = "Category",
       y = "Count") +
  scale_color_manual(values = c("Correct" = "green", "Incorrect" = "red")) +
  theme_minimal()