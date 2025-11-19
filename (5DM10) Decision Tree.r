#### Persiapan Data
# Importing the dataset
dataset = Social_Network_Ads
dataset = dataset[3:5]

# Encoding target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Analisis Deskriptif Data
summary(dataset)


#### Membagi Data Menjadi Data Training dan Testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling (optional untuk tree, tapi bisa tetap dilakukan)
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


#### 1. Model Decision Tree
# install.packages('rpart')
# install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

# Membuat model Decision Tree
dt_model = rpart(
  formula = Purchased ~ ., 
  data = training_set, 
  method = "class"
)

# Visualisasi pohon
rpart.plot(dt_model, type = 2, extra = 104, fallen.leaves = TRUE, main = "Decision Tree Model")


#### Prediksi
# a. Prediksi pada data training
y_pred_dt_train = predict(dt_model, newdata = training_set[-3], type = "class")

# b. Prediksi pada data testing
y_pred_dt_test = predict(dt_model, newdata = test_set[-3], type = "class")

# Confusion Matrix
cm_train_dt = table(Actual = training_set[, 3], Predicted = y_pred_dt_train)
cm_test_dt  = table(Actual = test_set[, 3], Predicted = y_pred_dt_test)

print(cm_train_dt)
print(cm_test_dt)


#### Evaluasi Model
library(MLmetrics)

# Fungsi bantu normalisasi label
as_lbl01 <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.numeric(x)) x <- as.character(x)
  factor(x, levels = c("0","1")) |> as.character()
}

# Fungsi evaluasi model
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
  return(hasil)
}

# Evaluasi hasil
evaluasi_model(y_pred_dt_train, training_set[, 3])
evaluasi_model(y_pred_dt_test,  test_set[, 3])


#### Visualisasi ROC & AUC
library(pROC)
library(ggplot2)

# Probabilitas prediksi (bukan class)
prob_pred_train_dt = predict(dt_model, newdata = training_set[-3], type = "prob")[,2]
prob_pred_test_dt  = predict(dt_model, newdata = test_set[-3], type = "prob")[,2]

y_train <- as.numeric(as.character(training_set$Purchased))
y_test  <- as.numeric(as.character(test_set$Purchased))

# ROC objects
roc_tr_dt <- roc(response = y_train, predictor = prob_pred_train_dt, quiet = TRUE)
roc_te_dt <- roc(response = y_test,  predictor = prob_pred_test_dt, quiet = TRUE)

# Plot ROC (overlay)
p_dt <- ggroc(list(Train = roc_tr_dt, Test = roc_te_dt), size = 1.2) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title = sprintf("ROC – Decision Tree | AUC Train = %.3f, Test = %.3f",
                    auc(roc_tr_dt), auc(roc_te_dt)),
    x = "1 - Spesifisitas", y = "Sensitivitas", color = "Set"
  ) +
  theme_minimal()

print(p_dt)

# Titik threshold 0.5
thr <- 0.5
tpr_dt <- sum(prob_pred_test_dt >= thr & y_test == 1) / sum(y_test == 1)
fpr_dt <- sum(prob_pred_test_dt >= thr & y_test == 0) / sum(y_test == 0)
p_dt + geom_point(aes(x = fpr_dt, y = tpr_dt), size = 3)
