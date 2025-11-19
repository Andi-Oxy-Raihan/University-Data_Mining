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

#### 2. Model Naive Bayes
library(e1071)
model_nb = naiveBayes(x = training_set[-3],
                        y = training_set$Purchased)

# Predicting
#a. predict data training
y_pred_nb_tr = predict(model_nb, newdata = training_set[-3])
# Making the Confusion Matrix
cm_train = table(training_set[, 3], y_pred_nb_tr)

#b. predicting data testing
y_pred_nb_test = predict(model_nb, newdata = test_set[-3])

# Making the Confusion Matrix
cm_testing = table(test_set[, 3], y_pred_nb_test)

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
evaluasi_model(training_set[, 3], y_pred_nb_tr)
evaluasi_model(test_set[, 3], y_pred_nb_test)

#### Menyelidiki overfitting atau underfitting
library(MLmetrics)

# Probabilitas posterior kelas "1"
prob_nb_tr   <- predict(model_nb, newdata = training_set[-3], type = "raw")[, "1"]
prob_nb_test <- predict(model_nb, newdata = test_set[-3],     type = "raw")[, "1"]

as_lbl01 <- function(x){ if(is.factor(x)) x <- as.character(x)
if(is.numeric(x)) x <- as.character(x)
factor(x, levels=c("0","1")) |> as.character() }
as_num01 <- function(x){ as.numeric(as_lbl01(x)) }

eval_full <- function(y_true, y_pred_lbl, y_prob){
  yt_ch <- as_lbl01(y_true); yp_ch <- as_lbl01(y_pred_lbl); yt_n <- as_num01(y_true)
  c(Accuracy  = Accuracy(yp_ch, yt_ch),
    F1        = F1_Score(yp_ch, yt_ch, positive="1"),
    AUC       = AUC(y_pred = y_prob, y_true = yt_n),
    LogLoss   = LogLoss(y_pred = y_prob, y_true = yt_n))
}

train_metrics <- eval_full(training_set$Purchased, y_pred_nb_tr,   prob_nb_tr)
test_metrics  <- eval_full(test_set$Purchased,     y_pred_nb_test, prob_nb_test)

gap <- test_metrics - train_metrics
round(rbind(Train=train_metrics, Test=test_metrics, Gap=test_metrics-train_metrics), 4)

## Interpretasi cepat:
## - Gap besar (Test << Train) → indikasi overfitting
## - Keduanya rendah → indikasi underfitting
