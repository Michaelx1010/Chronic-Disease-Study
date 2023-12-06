# Loading required packages
library(tidyverse)
library(ggplot2)
library(plotly)
library(ggthemes)
library(DT)
library(wesanderson)
library(ggpubr)
library(ROCR)
library(NHANES)
library(devtools)
library(data.table)
library(fpc)

# Load the data
us_chronic <- read_csv("data/us_chronic.csv")
datatable(head(us_chronic))

# Checking for time span of this data set
min(us_chronic$YearStart)
max(us_chronic$YearEnd)

# Checking for different data types
unique(us_chronic$DataValueType)

# Checking for location information
loc <- unique(us_chronic$LocationDesc)
loc

# Checking for different categories of Topic and Question of chronic diseases
types_t <- unique(us_chronic$Topic)
types_t

# Loop through each unique topic
for (topic in types_t) {
  cat("Topic:", topic, "\n")
  
  # Get unique questions for the current topic
  questions <- unique(us_chronic$Question[us_chronic$Topic == topic])
  
  # Print unique questions
  cat("Questions:\n", paste(questions, collapse = "\n"), "\n\n")
}

unique(us_chronic$Stratification1)

# Year summary
datatable(summary(us_chronic[, c("YearStart", "YearEnd")]))

# Data value summary for each distinct data types
us_chronic %>%
  group_by(DataValueType) %>%
  summarize(
    Mean = mean(DataValueAlt, na.rm = TRUE),
    Median = median(DataValueAlt, na.rm = TRUE),
    SD = sd(DataValueAlt, na.rm = TRUE),
    Min = min(DataValueAlt, na.rm = TRUE),
    Max = max(DataValueAlt, na.rm = TRUE)
  )

# Chronic diseases type visualization
p1 <- ggplot(us_chronic, aes(x = Topic, fill = Topic)) +
  geom_bar() +
  labs(title = "Frequency Distribution of Chronic diseases types",
       x = "Chronic diseases",
       y = "Frequency")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_hue()
ggplotly(p1)

# Locations visualization
p2 <- ggplot(us_chronic, aes(x = LocationDesc, fill = LocationDesc)) +
  geom_bar() +
  labs(title = "Frequency Distribution of Locations",
       x = "Locations",
       y = "Frequency")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  scale_fill_hue()
ggplotly(p2)

# Correlation analysis for Diabetes and Obesity
e1 <- us_chronic %>%
  filter(
    Question %in% c("Prevalence of diagnosed diabetes among adults aged >= 18 years", "Obesity among adults aged >= 18 years"),
    DataValueType == "Crude Prevalence",
    !is.na(Overall)
  )
e1 <- e1 %>%
  pivot_wider(names_from = Topic, values_from = DataValueAlt)
colnames(e1)[ncol(e1) - 1] <- "Obesity"
e1 <- e1 %>%
  select(YearStart, LocationAbbr, Diabetes, Obesity)
e1_filled <- e1 %>%
  group_by(YearStart, LocationAbbr) %>%
  mutate(Diabetes = ifelse(is.na(Diabetes), mean(Diabetes, na.rm = TRUE), Diabetes)) %>%
  ungroup() %>%
  arrange(YearStart, LocationAbbr)
e1_filled <- na.omit(e1_filled)
ep1 <- ggplot(e1_filled, aes(x = Obesity, y = Diabetes, color = LocationAbbr)) +
  geom_point() +
  labs(title = "Correlation between Diabetes and Obesity",
       x = "Obesity Prevalence (%) per state",
       y = "Diabetes Prevalence (%) per state") +
  theme_minimal()
ggplotly(ep1)

# Correlation analysis for High cholesterol and Tobacco use
e2 <- us_chronic %>%
  filter(
    Question %in% c("High cholesterol prevalence among adults aged >= 18 years", "Current smoking among adults aged >= 18 years"),
    DataValueType == "Crude Prevalence",
    !is.na(Overall)
  )
e2 <- e2 %>%
  pivot_wider(names_from = Topic, values_from = DataValueAlt)
colnames(e2)[ncol(e2) - 1] <- "Hc"
e2 <- e2 %>%
  select(YearStart, LocationAbbr, Hc, Tobacco)
e2_filled <- e2 %>%
  group_by(YearStart, LocationAbbr) %>%
  mutate(Hc = ifelse(is.na(Hc), mean(Hc, na.rm = TRUE), Hc)) %>%
  ungroup() %>%
  arrange(YearStart, LocationAbbr)
e2_filled <- na.omit(e2_filled)
ep2 <- ggplot(e2_filled, aes(x = Tobacco, y = Hc, color = LocationAbbr)) +
  geom_point() +
  labs(title = "Correlation between High cholesterol prevalence and Tobacco use",
       x = "Tobacco use (%) per state",
       y = "High cholesterol prevalence (%) per state") +
  theme_minimal()
ggplotly(ep2)

# Correlation analysis for Binge Drinking and Prostate Cancer mortality
e3 <- us_chronic %>%
  filter(
    Question %in% c("Cancer of the prostate, mortality", "Binge drinking frequency among adults aged >= 18 years who binge drink"),
    DataValueType %in% c( "Average Annual Crude Rate","Mean"),
    !is.na(Overall)
  )
e3 <- e3 %>%
  pivot_wider(names_from = Topic, values_from = DataValueAlt)
colnames(e3)[ncol(e3) - 1] <- "ProstateCancer"
e3 <- e3 %>%
  select(YearStart, LocationAbbr, ProstateCancer, Alcohol)
e3_filled <- e3 %>%
  group_by(YearStart, LocationAbbr) %>%
  mutate(ProstateCancer = ifelse(is.na(ProstateCancer), mean(ProstateCancer, na.rm = TRUE), ProstateCancer)) %>%
  ungroup() %>%
  arrange(YearStart, LocationAbbr)
e3_filled <- na.omit(e3_filled)
ep3 <- ggplot(e3_filled, aes(x = Alcohol, y = ProstateCancer, color = LocationAbbr)) +
  geom_point() +
  labs(title = "Correlation between Binge Drinking and Prostate Cancer mortality",
       x = "Means of binge drinking per state",
       y = "Prostate Cancer mortality (%) per state") +
  theme_minimal()
ggplotly(ep3)

# Histogram for Diabetes data
ggplot(e1_filled, aes(x = Diabetes)) +
  geom_histogram() +
  labs(title = "Histogram of Diabetes data", x = "Diabetes") +
  theme_minimal()

# Boxplot for Diabetes data
ggplot(e1_filled, aes(y = Diabetes)) +
  geom_boxplot() +
  labs(title = "Boxplot of Data", y = "Diabetes") +
  theme_minimal()

# Q-Q plot for Diabetes data
qqnorm(e1_filled$Diabetes)
qqline(e1_filled$Diabetes, col = "red")



# Load the Framingham Heart Study dataset
data <- read_csv("data/frmgham2.csv")
datatable(head(data))

# Generate summary statistics of the data
datatable(summary(data))

# Transform BMI into categorical groups
setDT(data)
data[,BMIgroup:= cut(BMI, c(0,18,25,30,100),labels=c("underweight","normal","overweight","obese"))]
data[,table(BMIgroup)]

# Visualization: Overall distribution histograms for continuous variables
a1<-ggplot(data, aes(x=AGE,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs( x = "Age", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="Zissou1"))

a2<-ggplot(data, aes(x=AGE,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs(x = "Age", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="Zissou1"))

b1<-ggplot(data, aes(x=BMI,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs( x = "BMI Index", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

b2<-ggplot(data, aes(x=BMI,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs( x = "BMI Index", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

g1<-ggplot(data, aes(x=GLUCOSE,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs( x = "Glucose level", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="Zissou1"))

g2<-ggplot(data, aes(x=GLUCOSE,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs( x = "Glucose level", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="Zissou1"))

t1<-ggplot(data, aes(x=TOTCHOL,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs( x = "Serum Total Cholesterol", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

t2<-ggplot(data, aes(x=TOTCHOL,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs( x = "Serum Total Cholesterol", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

s1<-ggplot(data, aes(x=SYSBP,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs(x = "Systolic Blood Pressure", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

s2<-ggplot(data, aes(x=SYSBP,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs( x = "Systolic Blood Pressure", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

d1<-ggplot(data, aes(x=DIABP,fill=BMIgroup)) + geom_histogram()+theme_gdocs()+
labs( x = "Diastolic Blood Pressure", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

d2<-ggplot(data, aes(x=DIABP,fill=PREVHYP)) + geom_histogram()+theme_gdocs()+
labs( x = "Diastolic Blood Pressure", y = "Count")+scale_fill_manual(values=wes_palette(n=5, name="IsleofDogs1"))

figure1 <- ggarrange(a1,a2,b1,b2,g1,g2,t1,t2,s1,s2,d1,d2,
                    ncol =3, nrow = 4)
figure1

# Relation plots for DIABETES
q1<-ggplot(data, aes(BMI,DIABETES))+
stat_smooth(method='glm', method.args=list(family='binomial'))+theme_gdocs()+
labs(title ="DIABETES versus BMI", x = "BMI", y = "DIABETES index")+
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q2<-ggplot(data, aes(GLUCOSE,DIABETES))+
stat_smooth(method='glm', method.args=list(family='binomial'))+theme_gdocs()+
labs(title ="versus GLUCOSE", x = "GLUCOSE", y = "DIABETES index")+
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q3<-ggplot(data, aes(SYSBP,DIABETES))+
stat_smooth(method='glm', method.args=list(family='binomial'))+theme_gdocs()+
labs(title ="versus Systolic Blood Pressure", x = "Systolic Blood Pressure", y = "DIABETES index")+
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q4<-ggplot(data, aes(DIABP,DIABETES))+
stat_smooth(method='glm', method.args=list(family='binomial'))+theme_gdocs()+
labs(title ="versus Dastolic Blood Pressure", x = "Dastolic Blood Pressure", y = "DIABETES index")+
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q5<-ggplot(data, aes(PREVSTRK,DIABETES))+
stat_smooth(method='glm', method.args=list(family='binomial'))+theme_gdocs()+
labs(title ="versus Prevalent Stroke", x = "Prevalent Stroke", y = "DIABETES index")+
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

figure9 <- ggarrange(q1,q2,q3,q4,q5,
                    ncol =3, nrow = 2)
figure9

# Relation plots for cardiovascular diseases
q1<-ggplot(data, aes(BMI,CVD)) +
stat_smooth(method='glm', method.args=list(family='binomial')) + theme_gdocs() +
labs(title ="CVD versus BMI", x = "BMI", y = "CVD index") +
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q2<-ggplot(data, aes(GLUCOSE,CVD)) +
stat_smooth(method='glm', method.args=list(family='binomial')) + theme_gdocs() +
labs(title ="CVD versus GLUCOSE", x = "GLUCOSE", y = "CVD index") +
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q3<-ggplot(data, aes(SYSBP,CVD)) +
stat_smooth(method='glm', method.args=list(family='binomial')) + theme_gdocs() +
labs(title ="CVD versus Systolic Blood Pressure", x = "Systolic Blood Pressure", y = "CVD index") +
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q4<-ggplot(data, aes(DIABP,CVD)) +
stat_smooth(method='glm', method.args=list(family='binomial')) + theme_gdocs() +
labs(title ="CVD versus Diastolic Blood Pressure", x = "Diastolic Blood Pressure", y = "CVD index") +
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

q5<-ggplot(data, aes(PREVSTRK,CVD)) +
stat_smooth(method='glm', method.args=list(family='binomial')) + theme_gdocs() +
labs(title ="CVD versus Prevalent Stroke", x = "Prevalent Stroke", y = "CVD index") +
scale_color_manual(values=wes_palette(n=5, name="Zissou1"))

figure9 <- ggarrange(q1, q2, q3, q4, q5,
                     ncol = 3, nrow = 2)
figure9