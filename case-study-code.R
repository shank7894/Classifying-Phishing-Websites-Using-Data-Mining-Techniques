# Case Study
library(tidyverse)
library(psych)
library(caret)
library(FNN)
library(ISLR)
library(tree)
library(randomForest)
library(neuralnet)
library(ROCR)
library(e1071)
library(gains)
library(ggplot2)
library(reshape2)
library(rpart)
library(rpart.plot)
library(corrplot)

# Load phishing_websites.csv
df <- data.frame(read.csv("./data/phishing_websites.csv"))
# Remove "HttpsInHostname" column because it contains a few NAs
df$PHISHING_WEBSITE <- as.factor(ifelse(df$CLASS_LABEL == 1, "Yes", "No"))
df <- df[, !colnames(df) %in% c("HttpsInHostname", "CLASS_LABEL")]

describe(df)
##############################################################
## Data Visualization

# Let's look at NumDots Histogram
ggplot(df, aes(NumDots)) +
  geom_histogram(binwidth = 1, color = "black", fill = "steelblue") +
  ggtitle("NumDots Histogram")

# Let's look at the UrlLength Histogram
ggplot(df, aes(UrlLength)) +
  geom_histogram(binwidth = 25, color = "black", fill = "steelblue") +
  ggtitle("UrlLength Histogram")

# Let's look at whether having an IP address in the Url gives us any information as
# to whether the website is a phishing website or not
ggplot(df, aes(as.factor(IpAddress), fill = PHISHING_WEBSITE)) +
  geom_histogram(stat = "count") +
  ggtitle("IpAddress Barplot") +
  labs(x = "IP Address")

# corrplot

cor <- round(cor(df[, 1:47]), 2)
corrplot(cor, type = "upper")
## Looks like all sites having an IP Address in the Url are phishing websites.

##############################################################
## Data Pre-processing

# Define the normalize function
normalize <- function(x) {
  return((x - min(x))) / (max(x) - min(x))
}

# Normalize the data frame
df.norm <- as.data.frame(cbind(
  as.data.frame(lapply(df[1:47], normalize)),
  df$PHISHING_WEBSITE
)) %>%
  rename(PHISHING_WEBSITE = "df$PHISHING_WEBSITE")


##############################################################
## Data Reduction and Transformation


# Performing PCA on the data
# Perform Scree Plot and Parallel Analysis
fa.parallel(df.norm[, 1:47], fa = "pc", n.iter = 100, show.legend = FALSE)

# Perform PCA with 13 components
pc <- principal(df.norm[, 1:47], nfactors = 13, rotate = "none", scores = TRUE)
pc <- cbind(as.data.frame(pc$scores), df.norm$PHISHING_WEBSITE) %>%
  rename(PHISHING_WEBSITE = "df.norm$PHISHING_WEBSITE")

##############################################################
## Data Mining Techniques
# Splitting data into training and validation sets
# Generate the training data indices

set.seed(20)
indices <- sample(seq_len(nrow(pc)), size = floor(0.6 * nrow(pc)))
# Get training and validation data
train_data <- pc[indices, ]
validation_data <- pc[-indices, ]

levels(train_data$PHISHING_WEBSITE) <-
  make.names(levels(factor(train_data$PHISHING_WEBSITE)))
levels(validation_data$PHISHING_WEBSITE) <-
  make.names(levels(factor(validation_data$PHISHING_WEBSITE)))

# corrplot of pca data
cor <- cor(pc[, 1:13])
corrplot(cor, type = "upper")

# Also keep a set of train and validation sets without PCA
df.norm.train <- as.data.frame(lapply(df.norm[indices, ], as.numeric))
df.norm.validation <- as.data.frame(lapply(df.norm[-indices, ], as.numeric))

df.norm.train <- df.norm[indices, ]
df.norm.validation <- df.norm[-indices, ]
df.norm.train$PHISHING_WEBSITE <- as.factor(df.norm.train$PHISHING_WEBSITE)
df.norm.validation$PHISHING_WEBSITE <- as.factor(df.norm.validation$PHISHING_WEBSITE)

levels(df.norm.train$PHISHING_WEBSITE) <-
  make.names(levels(factor(df.norm.train$PHISHING_WEBSITE)))
levels(df.norm.validation$PHISHING_WEBSITE) <-
  make.names(levels(factor(df.norm.validation$PHISHING_WEBSITE)))



# Creating a performance list for each algorithm
performance_list <- data.frame(
  "Model" = character(),
  "AUC" = numeric(),
  "Accuracy" = numeric()
)

model_names <- list()
lift_charts <- list()
roc_curves <- list()

# Helper Function to plot ROC Curve and Calculate Accuracy

evaluate_performance <- function(pred, labels, model_name) {
  model_names[[length(model_names) + 1]] <<- model_name
  
  # Accuracy
  pred.class <- ifelse(slot(pred, "predictions")[[1]] > 0.5, "Yes", "No")
  levels(pred.class) <- make.names(levels(factor(pred.class)))

  acc <- confusionMatrix(table(pred.class, labels))$overall[[1]] * 100

  # ROC Plot
  roc <- performance(pred, "tpr", "fpr")
  plot(roc, col = "red", lwd = 2, main = paste0(model_name, " ROC Curve"))
  abline(a = 0, b = 1)
  
  roc_curves[[length(roc_curves) + 1]] <<- roc


  auc <- performance(pred, measure = "auc")

  temp <- data.frame(
    "Model" = model_name,
    "AUC" = auc@y.values[[1]],
    "Accuracy" = acc
  )
  performance_list <<- rbind(performance_list, temp)
  print("Updated Performance List")

  lift <- performance(pred, "tpr", "rpp")
  plot(lift, main = paste0(model_name, " Lift Curve"), col = "green")
  abline(a = 0, b = 1)

  lift_charts[[length(lift_charts) + 1]] <<- lift

  rm(list = c("auc", "acc", "roc", "pred.class", "temp", "lift"))
}


#######################
## Implementing KNN

# Setting up train controls
tc <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


set.seed(20)
knn.model <- train(PHISHING_WEBSITE ~ .,
  data = train_data, method = "knn",
  trControl = tc,
  metric = "ROC",
  tuneLength = 10
)

# Look at the KNN Model
knn.model
plot(knn.model)

# get predictions for validation data
knn.pred <- predict(knn.model, validation_data, type = "prob")
pred.val <- prediction(knn.pred[, 2], validation_data$PHISHING_WEBSITE)


evaluate_performance(pred.val, validation_data$PHISHING_WEBSITE, "KNN")

rm(list = c("knn.model", "tc", "knn.pred", "pred.val"))


#######################
## Implementing Logistic Regression
# On PCA Dataset
set.seed(20)
glm.fit.pc <- glm(PHISHING_WEBSITE ~ ., data = train_data, family = binomial)

glm.probs.pc <- predict(glm.fit.pc, newdata = validation_data, type = "response")
pred.val <- prediction(glm.probs.pc, validation_data$PHISHING_WEBSITE)

evaluate_performance(
  pred.val, validation_data$PHISHING_WEBSITE,
  "Logistic Regression (PCA)"
)


# On Original Dataset
set.seed(20)
glm.fit <- glm(PHISHING_WEBSITE ~ ., data = df.norm.train, family = binomial)

glm.probs <- predict(glm.fit, newdata = df.norm.validation, type = "response")
pred.val <- prediction(glm.probs, df.norm.validation$PHISHING_WEBSITE)

evaluate_performance(
  pred.val, validation_data$PHISHING_WEBSITE,
  "Logistic Regression"
)

rm(list = c(
  "glm.fit", "glm.probs",
  "glm.fit.pc", "glm.probs.pc",
  "pred.val"
))

#######################
## Implementing Naive Bayes
set.seed(20)
nb <- naiveBayes(PHISHING_WEBSITE ~ ., data = train_data)

nb.pred <- predict(nb, newdata = validation_data, type = "raw")
pred.val <- prediction(nb.pred[, 2], validation_data$PHISHING_WEBSITE)

evaluate_performance(pred.val, validation_data$PHISHING_WEBSITE, "Naive Bayes (PCA)")

rm(list = c("nb", "nb.pred", "pred.val"))

#######################
## Implementing Decision Tree
# Classification tree on PCA Dataset

set.seed(20)
tree.pca <- rpart(PHISHING_WEBSITE ~ ., data = train_data, method = "class")
rpart.plot(tree.pca, main = "Classification Tree (PCA)")
tree.pca.pred <- predict(tree.pca, validation_data)
pred.val <- prediction(tree.pca.pred[, 2], validation_data$PHISHING_WEBSITE)


evaluate_performance(
  pred.val,
  validation_data$PHISHING_WEBSITE,
  "Classification Tree (PCA)"
)

# Classification tree on Original Dataset
set.seed(20)
tree <- rpart(PHISHING_WEBSITE ~ ., data = df.norm.train, method = "class")
rpart.plot(tree, main = "Classification Tree")

print(tree$variable.importance)

tree.pred <- predict(tree, df.norm.validation)
pred.val <- prediction(tree.pred[, 2], df.norm.validation$PHISHING_WEBSITE)

evaluate_performance(
  pred.val,
  validation_data$PHISHING_WEBSITE,
  "Classification Tree"
)

rm(list = c(
  "tree.pca", "tree.pca.pred", "tree",
  "tree.pred", "pred.val"
))

#######################
## Implementing Random Forests

# On PCA dataset
set.seed(20)
rf.pca <- randomForest(PHISHING_WEBSITE ~ ., data = train_data)
rf.pca.pred <- predict(rf.pca, validation_data, type = "prob")
pred.val <- prediction(rf.pca.pred[, 2], validation_data$PHISHING_WEBSITE)

evaluate_performance(
  pred.val,
  validation_data$PHISHING_WEBSITE,
  "Random Forest (PCA)"
)
# On original dataset
set.seed(20)
rf <- randomForest(PHISHING_WEBSITE ~ ., data = df.norm.train)
rf.pred <- predict(rf, df.norm.validation, type = "prob")
pred.val <- prediction(rf.pred[, 2], df.norm.validation$PHISHING_WEBSITE)

evaluate_performance(
  pred.val,
  df.norm.validation$PHISHING_WEBSITE,
  "Random Forest"
)


rm(list = c(
  "rf.pca", "rf.pca.pred", "rf",
  "rf.pred", "pred.val"
))

#######################
## Implementing Support Vector Machine
tc <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(20)
svm.pca <- train(PHISHING_WEBSITE ~ ., train_data,
  method = "svmLinear",
  trControl = tc, tuneLength = 10
)

svm.pca.pred <- predict(svm.pca, validation_data, type = "prob")

pred.val <- prediction(svm.pca.pred[, 2], validation_data$PHISHING_WEBSITE)

evaluate_performance(
  pred.val,
  validation_data$PHISHING_WEBSITE,
  "SVM (PCA)"
)

# On Original Dataset
tc <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(20)
svm <- train(PHISHING_WEBSITE ~ .,
  df.norm.train,
  method = "svmLinear",
  preProcess = NULL,
  trControl = tc,
  metric = "ROC",
  tuneLength = 10
)

svm.pred <- predict(svm, df.norm.validation, type = "prob")

pred.val <- prediction(svm.pred[, 2], df.norm.validation$PHISHING_WEBSITE)

evaluate_performance(
  pred.val,
  validation_data$PHISHING_WEBSITE,
  "SVM"
)

rm(list = c("tc", "svm.pca", "svm", "svm.pred", "svm.pca.pred", "pred.val"))
#######################
## Implementing Artificial Neural Networks
# On PCA Dataset
set.seed(20)
nn.pca <- neuralnet(PHISHING_WEBSITE ~ .,
  data = train_data,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE
)

plot(nn.pca, main = "Artificial Neural Net (PCA)")

nn.pca.pred <- neuralnet::compute(nn.pca, validation_data[, 1:13])$net.result
pred.val <- prediction(nn.pca.pred[, 2], validation_data$PHISHING_WEBSITE)
evaluate_performance(
  pred.val, validation_data$PHISHING_WEBSITE,
  "Artificial Neural Net (PCA)"
)

rm(list = c("nn.pca", "nn.pca.pred", "pred.val"))
##############################################################

write.csv(performance_list, "performance_list.csv")

# Plot all ROC Curves
par(mfrow = c(3, 4))
for(i in 1:length(roc_curves)) {
  plot(roc_curves[[i]], main = model_names[[i]], col = "red")
  abline(a = 0, b = 1)
}


# Plot all lift charts
par(mfrow = c(3, 4))
for(i in 1:length(lift_charts)) {
  plot(lift_charts[[i]], main = model_names[[i]], col = "green")
  abline(a = 0, b = 1)
}


