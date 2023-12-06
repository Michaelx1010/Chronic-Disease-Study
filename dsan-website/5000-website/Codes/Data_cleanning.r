# Load necessary libraries
library(tidyverse)
library(DT)

# Load the raw data
us_chronic <- read_csv("data/U.S._Chronic_Disease_Indicators__CDI_.csv")

# Display the first few rows of the data using datatable
a <- head(us_chronic)
datatable(a)

# Checking for different chronic disease types
# Convert 'Topic' column to a factor
us_chronic$Topic <- as.factor(us_chronic$Topic)

# Get the unique categories of different disease incidents
category <- levels(us_chronic$Topic)
category

# Checking for NA columns
# Check for NA values in each column
na_columns <- colSums(is.na(us_chronic))

# Display columns with NA values
print(na_columns)

# Drop the columns with completely NA values
us_chronic <- us_chronic[, colSums(is.na(us_chronic)) != nrow(us_chronic)]
datatable(head(us_chronic))

# Transforming data to tidy format

# Investigate the unique values in 'StratificationCategory1' and 'Stratification1'
types_s <- unique(us_chronic$StratificationCategory1)
for (i in types_s) {
  cat("Category:", i, "\n")
  
  # Get unique Types for the current category
  types <- unique(us_chronic$Stratification1[us_chronic$StratificationCategory1 == i])
  
  # Print unique types
  cat("Types:\n", paste(types, collapse = "\n"), "\n\n")
}

# Create new columns 'Race', 'Gender', and 'Overall'
us_chronic_tidy <- us_chronic %>%
  mutate(Race = ifelse(StratificationCategory1 == "Race/Ethnicity", Stratification1, NA),
         Gender = ifelse(StratificationCategory1 == "Gender", Stratification1, NA),
         Overall = ifelse(StratificationCategory1 == "Overall", Stratification1, NA)) %>%
  
  select(-StratificationCategory1, -Stratification1) %>%
  
  mutate(RaceID = ifelse(StratificationCategoryID1 == "RACE", StratificationID1, NA),
         GenderID = ifelse(StratificationCategoryID1 == "GENDER", StratificationID1, NA),
         OverallID = ifelse(StratificationCategoryID1 == "OVERALL", StratificationID1, NA)) %>%
  
  select(-StratificationCategoryID1, -StratificationID1)

# View the resulting tidy dataset
datatable(head(us_chronic_tidy))

# Save the cleaned data
write.csv(us_chronic_tidy, file = 'data/us_chronic.csv', row.names = FALSE)