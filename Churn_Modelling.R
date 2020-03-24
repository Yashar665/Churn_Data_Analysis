# Import libraries & dataset ----

library(tidyverse)
library(data.table)
library(caTools)
library(inspectdf)
library(Hmisc)
library(e1071)
library(ROCR)
library(Amelia)

data <- fread('C:/Users/Yashar/Desktop/Data Science Bootcamp/R programming/Week 7/Churn_Modelling.csv',
              na.strings = "")


# Filling NAs ----

data %>% inspect_na()
missmap(data,col=c("blue", "red"), main = "Missing values vs observed") #same with inspect_na but visual


data %>% glimpse()

data <- data %>% rename(ID = `CustomerId`)

data <- data %>% select(-Surname)

data$Exited <- data$Exited %>% as_factor()

data$Gender <- data$Gender %>% as_factor() %>% as.numeric()

data$Geography <- data$Geography %>% as_factor() %>% as.numeric()

data %>% glimpse()
# Modeling ----

df <- data %>% select(-ID)

set.seed(123)
split <- df$Exited %>% sample.split(SplitRatio = 0.8)
train <- df %>% subset(split == T)
test <- df %>% subset(split == F)

# Feature Scaling
train <- train[,-12] %>% scale() %>% as.data.frame() %>% cbind(Exited=train$Exited)
test <- test[,-12] %>% scale() %>% as.data.frame() %>% cbind(Exited=test$Exited)


# Fitting SVM ----
model <- svm(formula = Exited~.,
             data = train,
             type = 'C-classification',
             probability = T,
             kernel = 'linear') #'linear','sigmoid','polynomial'

# Predicting the Test set results for SVM
pred <- model %>% predict(test %>% select(-Exited),
                          probability=T)

prob <- attr(pred,"probabilities")


# Evaluation Metrices (Accuracy & AUC) for SVM ----

train_label <- train %>% pull(Exited)
test_label <- test %>% pull(Exited)
nc <- train_label %>% unique() %>% length()

p <- prob %>% 
  as.data.frame() %>% 
  mutate(label = test_label) %>% 
  bind_cols(pred = as.data.frame(pred))

# Accuracy
cm <- table(Prediction = p$pred, Actual = p$label)
tp <- cm[4]
tn <- cm[1]
fn <- cm[3]
fp <- cm[2]

accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
F1_score <- 2 * (recall * precision) / (recall + precision)

tibble(accuracy,
       F1_score)


pred_obj <- prediction(p$`1`,test$Exited)

# # Treshold
# eval <- pred_obj %>% performance("acc")
# Treshold <- eval %>% slot("x.values") %>% .[[1]] %>% .[-1] %>% max() %>% round(2)
# 
# # ROC Curve
# roc <- pred_obj %>% performance("tpr", "fpr")
# roc %>% plot(colorize=T,
#              main = "ROC Curve",
#              ylab = "Sensitivity",
#              xlab = "1-Specificity")
# abline(a=0, b=1)

# AUC
auc <- pred_obj %>% performance("auc")
auc_test <- auc %>% slot("y.values") %>% .[[1]] %>% round(2)


# Check overfitting for SVM ----

pred <- model %>% predict(train %>% select(-Exited),
                          probability=T)

prob <- attr(pred,"probabilities")

p <- prob %>% 
  as.data.frame() %>% 
  mutate(label = train_label) %>% 
  bind_cols(pred = as.data.frame(pred))

pred_obj <- prediction(p$`1`,train$Exited)

auc <- pred_obj %>% performance("auc")
auc_train <- auc %>% slot("y.values") %>% .[[1]] %>% round(2)

tibble(auc_train,
       auc_test)


# Fitting Naive ----
model <- naiveBayes(x = train[-12],
                    y = train$Exited)

# Predicting the Test set results for Naive
pred <- model %>% predict(test %>% select(-Exited))

prob <- model %>% predict(test %>% select(-Exited),
                          type = "raw")


# Evaluation Metrices (Accuracy & AUC) for Naive ----

train_label <- train %>% pull(Exited)
test_label <- test %>% pull(Exited)
nc <- train_label %>% unique() %>% length()

p <- prob %>% 
  as.data.frame() %>% 
  mutate(label = test_label) %>% 
  bind_cols(pred = as.data.frame(pred))

# Accuracy
cm <- table(Prediction = p$pred, Actual = p$label)
tp <- cm[4]
tn <- cm[1]
fn <- cm[3]
fp <- cm[2]

accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)
F1_score <- 2 * (recall * precision) / (recall + precision)

tibble(accuracy,
       F1_score)


pred_obj <- prediction(p$`1`,test$Exited)

# # Treshold
# eval <- pred_obj %>% performance("acc")
# Treshold <- eval %>% slot("x.values") %>% .[[1]] %>% .[-1] %>% max() %>% round(2)
# 
# # ROC Curve
# roc <- pred_obj %>% performance("tpr", "fpr")
# roc %>% plot(colorize=T,
#              main = "ROC Curve",
#              ylab = "Sensitivity",
#              xlab = "1-Specivity")
# abline(a=0, b=1)

# AUC
auc <- pred_obj %>% performance("auc")
auc_test <- auc %>% slot("y.values") %>% .[[1]] %>% round(2)


# Check overfitting for Naive ----

pred <- model %>% predict(train %>% select(-Exited))

prob <- model %>% predict(train %>% select(-Exited),
                          type = "raw")

p <- prob %>%
  as.data.frame() %>%
  mutate(label = train_label) %>%
  bind_cols(pred = as.data.frame(pred))

pred_obj <- prediction(p$`1`,train$Exited)

auc <- pred_obj %>% performance("auc")
auc_train <- auc %>% slot("y.values") %>% .[[1]] %>% round(2)

tibble(auc_train,
       auc_test)


#Fitting Logistic regression----
model <- glm(Exited ~.,family=binomial,data=train)



#Assessing the predictive ability of the model
fitted.results <- predict(model,newdata = subset(test,select=1:11),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Exited,na.rm = T)
print(paste('Accuracy',1-misClasificError))


#ROC
#install.packages('ROCR')
library(ROCR)
p <- predict(model, newdata = subset(test,select=c(1,2,3,4,5,6,7,8,9,10,11)), type="response")
pr <- prediction(p, test$Exited)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf,colorize=TRUE)



auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc



# Best model out of 3 models(SVM, Naive Bayes, and Logistic Regression) is Naive Bayes
# Because AUC for SVM is 0.68, 0.69 for train and test set respectively
#         AUC for Naive Bayes is 0.8, 0.82 for train and test set respectively
#         AUC for Logistic Regression is 0.81, 0.77 for train and test set respectively