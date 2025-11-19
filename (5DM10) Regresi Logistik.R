#### Persiapan Data
# Importing the dataset
dataset = Social_Network_Ads
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Analisis Deskriptif Data
summary(dataset)


####Membagi Data Menjadi Data Training dan Testing
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

#### 1. Model Regresi Logistik
# Fitting Logistic Regression to the Training set
lr_model = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting
#a. predict data training
prob_pred_train = predict(lr_model, type = 'response', 
                          newdata = training_set[-3])
y_pred_lr_train = ifelse(prob_pred_train > 0.5, 1, 0)

# Making the Confusion Matrix
cm_train = table(training_set[, 3], y_pred_lr_train)

#b. predicting data testing
prob_pred_testing = predict(lr_model, type = 'response', newdata = test_set[-3])
y_pred_lr_test = ifelse(prob_pred_testing > 0.5, 1, 0)

# Making the Confusion Matrix
cm_testing = table(test_set[, 3], y_pred_lr_test)

####Evaluasi Model
#install.packages("MLmetrics")
library("MLmetrics")
#Membuat function evaluasi
# install.packages("MLmetrics") # jika belum ada
library(MLmetrics)

# Normalisasi label ke "0"/"1" (karakter)
as_lbl01 <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.numeric(x)) x <- as.character(x)
  factor(x, levels = c("0","1")) |> as.character()
}

evaluasi_model <- function(pred, aktual, positive = "1") {
  pred   <- as_lbl01(pred)
  aktual <- as_lbl01(aktual)
  
  hasil <- c(
    akurasi      = Accuracy(pred, aktual),
    presisi      = Precision(pred, aktual,  positive = positive),
    recall       = Recall(pred, aktual,     positive = positive),
    spesifisitas = Specificity(pred, aktual,positive = positive),
    f1           = F1_Score(pred, aktual,   positive = positive)
  )
  return(hasil)  # bisa juga return(as.list(hasil)) bila ingin list
}

#evaluasi prediksi
evaluasi_model(training_set[, 3], y_pred_lr_train)
evaluasi_model(test_set[, 3], y_pred_lr_test)

#Visualisasi ROC
# install.packages("pROC")    # jika belum ada
library(pROC)
library(ggplot2)

# Ubah label ke numerik 0/1 (Purchased punya level c(0,1))
y_train <- as.numeric(as.character(training_set$Purchased))
y_test  <- as.numeric(as.character(test_set$Purchased))

# ROC objects
roc_tr <- roc(response = y_train, predictor = prob_pred_train, quiet = TRUE)
roc_te <- roc(response = y_test,  predictor = prob_pred_testing, quiet = TRUE)

# Plot ROC (overlay) + AUC
p <- ggroc(list(Train = roc_tr, Test = roc_te), size = 1.2) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title = sprintf("ROC – Logistic Regression | AUC Train = %.3f, Test = %.3f",
                    auc(roc_tr), auc(roc_te)),
    x = "1 - Spesifisitas", y = "Sensitivitas", color = "Set"
  ) +
  theme_minimal()

print(p)

thr <- 0.5
tpr <- sum(prob_pred_testing >= thr & y_test == 1) / sum(y_test == 1)
fpr <- sum(prob_pred_testing >= thr & y_test == 0) / sum(y_test == 0)
p + geom_point(aes(x = fpr, y = tpr), size = 3)
