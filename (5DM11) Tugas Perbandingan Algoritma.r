#Script R Studio Khusus Tugas Dataset "Wine"
# Naive Bayes
dataset <- data.tugas.1.Wine

# Pastikan kolom target ada
if(!"Customer_Segment" %in% names(dataset)) {
  stop("Kolom 'Customer_Segment' tidak ditemukan. Pastikan nama kolom sama persis di CSV.")
}

# --- Pilih kolom fitur + target ---
# Sesuaikan: ambil semua kolom kecuali yang tidak relevan
# (Kalau dataset kamu sudah benar, biarkan semua kecuali Customer_Segment)
# Misalnya jika kolom 3:5 adalah fitur seperti di kode awal kamu:
dataset <- dataset[, c(3:5, which(names(dataset) == "Customer_Segment"))]

# Ubah target menjadi faktor
dataset$Customer_Segment <- factor(dataset$Customer_Segment, levels = unique(dataset$Customer_Segment))

# Analisis deskriptif
summary(dataset)

#### Membagi Data Menjadi Training dan Testing
if(!require(caTools)) install.packages("caTools", dependencies = TRUE)
library(caTools)
set.seed(123)
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling
training_set[, -ncol(training_set)] <- scale(training_set[, -ncol(training_set)])
test_set[, -ncol(test_set)] <- scale(test_set[, -ncol(test_set)])

#### 2. Model Naive Bayes
if(!require(e1071)) install.packages("e1071", dependencies = TRUE)
library(e1071)

# Jalankan model Naive Bayes berdasarkan data training
model_nb <- naiveBayes(Customer_Segment ~ ., data = training_set)

# Predicting
# a. prediksi data training
y_pred_nb_tr <- predict(model_nb, newdata = training_set[-ncol(training_set)])
cm_train <- table(Actual = training_set$Customer_Segment, Predicted = y_pred_nb_tr)
cat("\nConfusion Matrix (Training):\n")
print(cm_train)

# b. prediksi data testing
y_pred_nb_test <- predict(model_nb, newdata = test_set[-ncol(test_set)])
cm_testing <- table(Actual = test_set$Customer_Segment, Predicted = y_pred_nb_test)
cat("\nConfusion Matrix (Testing):\n")
print(cm_testing)

#### Evaluasi Model
if(!require(MLmetrics)) install.packages("MLmetrics", dependencies = TRUE)
library(MLmetrics)

# --- Fungsi bantu evaluasi ---
as_lbl01 <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  if (is.numeric(x)) x <- as.character(x)
  factor(x, levels = c("0","1")) |> as.character()
}

evaluasi_model <- function(pred, aktual, positive = "1") {
  pred   <- as_lbl01(pred)
  aktual <- as_lbl01(aktual)
  hasil <- c(
    Akurasi      = Accuracy(pred, aktual),
    Presisi      = Precision(pred, aktual,  positive = positive),
    Recall       = Recall(pred, aktual,     positive = positive),
    Spesifisitas = Specificity(pred, aktual,positive = positive),
    F1           = F1_Score(pred, aktual,   positive = positive)
  )
  return(round(hasil, 4))
}

cat("\nEvaluasi Data Training:\n")
print(evaluasi_model(y_pred_nb_tr, training_set$Customer_Segment))

cat("\nEvaluasi Data Testing:\n")
print(evaluasi_model(y_pred_nb_test, test_set$Customer_Segment))

#### Analisis Overfitting / Underfitting
# Probabilitas posterior kelas "1"
prob_nb_tr   <- predict(model_nb, newdata = training_set[-ncol(training_set)], type = "raw")[, "1"]
prob_nb_test <- predict(model_nb, newdata = test_set[-ncol(test_set)],     type = "raw")[, "1"]

as_num01 <- function(x) { as.numeric(as_lbl01(x)) }

eval_full <- function(y_true, y_pred_lbl, y_prob) {
  yt_ch <- as_lbl01(y_true)
  yp_ch <- as_lbl01(y_pred_lbl)
  yt_n  <- as_num01(y_true)
  c(
    Accuracy = Accuracy(yp_ch, yt_ch),
    F1       = F1_Score(yp_ch, yt_ch, positive = "1"),
    AUC      = AUC(y_pred = y_prob, y_true = yt_n),
    LogLoss  = LogLoss(y_pred = y_prob, y_true = yt_n)
  )
}

train_metrics <- eval_full(training_set$Customer_Segment, y_pred_nb_tr, prob_nb_tr)
test_metrics  <- eval_full(test_set$Customer_Segment, y_pred_nb_test, prob_nb_test)

cat("\nPerbandingan Kinerja (Train vs Test):\n")
print(round(rbind(Train = train_metrics,
                  Test  = test_metrics,
                  Gap   = test_metrics - train_metrics), 4))

cat("\nInterpretasi Cepat:\n")
cat("- Gap besar (Test << Train) → indikasi overfitting\n")
cat("- Keduanya rendah → indikasi underfitting\n")

#Regresi logistik
#### 1. Persiapan Data
# Gunakan dataset yang sudah ada di environment
dataset <- data.tugas.1.Wine

# Pastikan semua kolom faktor dikonversi ke numerik bila perlu
dataset[] <- lapply(dataset, function(x) {
  if (is.factor(x)) as.numeric(as.character(x)) else x
})

# Pastikan kolom target bertipe faktor
dataset$Customer_Segment <- as.factor(dataset$Customer_Segment)

# Analisis Deskriptif
summary(dataset)


#### 2. Membagi Data Menjadi Data Training dan Testing
library(caTools)
set.seed(123)
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature Scaling (selain kolom target)
training_set[-ncol(training_set)] <- scale(training_set[-ncol(training_set)])
test_set[-ncol(test_set)] <- scale(test_set[-ncol(test_set)])


#### 3. Model Regresi Logistik
# Fitting Logistic Regression ke data training
lr_model <- glm(formula = Customer_Segment ~ .,
                family = binomial,
                data = training_set)

# Prediksi
# a. Training
prob_pred_train <- predict(lr_model, type = 'response', newdata = training_set[-ncol(training_set)])
y_pred_lr_train <- ifelse(prob_pred_train > 0.5, 1, 0)

# Confusion Matrix Training
cm_train <- table(training_set[, ncol(training_set)], y_pred_lr_train)

# b. Testing
prob_pred_testing <- predict(lr_model, type = 'response', newdata = test_set[-ncol(test_set)])
y_pred_lr_test <- ifelse(prob_pred_testing > 0.5, 1, 0)

# Confusion Matrix Testing
cm_testing <- table(test_set[, ncol(test_set)], y_pred_lr_test)


#### 4. Evaluasi Model
library(MLmetrics)

# Fungsi bantu konversi label
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
  return(hasil)
}

# Evaluasi prediksi
cat("Evaluasi Training:\n")
print(evaluasi_model(training_set[, ncol(training_set)], y_pred_lr_train))
cat("\nEvaluasi Testing:\n")
print(evaluasi_model(test_set[, ncol(test_set)], y_pred_lr_test))


#### 5. Visualisasi ROC
library(pROC)
library(ggplot2)

# Konversi label ke numerik
y_train <- as.numeric(as.character(training_set$Customer_Segment))
y_test  <- as.numeric(as.character(test_set$Customer_Segment))

# ROC objects
roc_tr <- roc(response = y_train, predictor = prob_pred_train, quiet = TRUE)
roc_te <- roc(response = y_test,  predictor = prob_pred_testing, quiet = TRUE)

# Plot ROC
p <- ggroc(list(Train = roc_tr, Test = roc_te), size = 1.2) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
  labs(
    title = sprintf("ROC – Logistic Regression | AUC Train = %.3f, Test = %.3f",
                    auc(roc_tr), auc(roc_te)),
    x = "1 - Spesifisitas", y = "Sensitivitas", color = "Set"
  ) +
  theme_minimal()

print(p)

# Titik threshold 0.5
thr <- 0.5
tpr <- sum(prob_pred_testing >= thr & y_test == 1) / sum(y_test == 1)
fpr <- sum(prob_pred_testing >= thr & y_test == 0) / sum(y_test == 0)
p + geom_point(aes(x = fpr, y = tpr), size = 3)

#Decision Tree
#### Persiapan Data ----
# Pastikan file sudah ada di direktori kerja
dataset <- read.csv("data tugas 1 Wine.csv", stringsAsFactors = FALSE)

# Pastikan target adalah faktor (kelas)
dataset$Customer_Segment <- as.factor(dataset$Customer_Segment)

# Analisis deskriptif awal
summary(dataset)
str(dataset)


#### Membagi Data Menjadi Training dan Testing ----
library(caTools)
set.seed(123)
split <- sample.split(dataset$Customer_Segment, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Feature scaling (optional, biasanya tidak wajib untuk decision tree)
training_set[-ncol(training_set)] <- scale(training_set[-ncol(training_set)])
test_set[-ncol(test_set)] <- scale(test_set[-ncol(test_set)])


#### 1️⃣ Model Decision Tree ----
library(rpart)
library(rpart.plot)

# Membangun model decision tree
dt_model <- rpart(
  formula = Customer_Segment ~ .,
  data = training_set,
  method = "class"
)

# Visualisasi pohon
rpart.plot(dt_model, type = 2, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree Model")


#### Prediksi ----
# a. Prediksi pada data training
y_pred_train <- predict(dt_model, newdata = training_set[-ncol(training_set)], type = "class")

# b. Prediksi pada data testing
y_pred_test <- predict(dt_model, newdata = test_set[-ncol(test_set)], type = "class")

# Confusion Matrix
cm_train <- table(Actual = training_set$Customer_Segment, Predicted = y_pred_train)
cm_test  <- table(Actual = test_set$Customer_Segment, Predicted = y_pred_test)

print(cm_train)
print(cm_test)


#### Evaluasi Model ----
library(MLmetrics)

# Fungsi bantu
as_lbl01 <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  factor(x)
}

# Fungsi evaluasi
evaluasi_model <- function(pred, aktual) {
  pred   <- as_lbl01(pred)
  aktual <- as_lbl01(aktual)
  
  hasil <- c(
    Akurasi = Accuracy(pred, aktual),
    Presisi = Precision(pred, aktual, average = "macro"),
    Recall  = Recall(pred, aktual, average = "macro"),
    F1      = F1_Score(pred, aktual, average = "macro")
  )
  return(hasil)
}

# Evaluasi Training & Testing
hasil_train <- evaluasi_model(y_pred_train, training_set$Customer_Segment)
hasil_test  <- evaluasi_model(y_pred_test,  test_set$Customer_Segment)

print("Evaluasi Training:")
print(hasil_train)
print("Evaluasi Testing:")
print(hasil_test)


#### ROC & AUC ----
library(pROC)
library(ggplot2)

# Probabilitas prediksi
prob_pred_train <- predict(dt_model, newdata = training_set[-ncol(training_set)], type = "prob")
prob_pred_test  <- predict(dt_model, newdata = test_set[-ncol(test_set)], type = "prob")

# Karena multiclass, hitung AUC macro-average
roc_list_train <- multiclass.roc(training_set$Customer_Segment, prob_pred_train)
roc_list_test  <- multiclass.roc(test_set$Customer_Segment, prob_pred_test)

auc_train <- auc(roc_list_train)
auc_test  <- auc(roc_list_test)

cat("\nAUC Training:", auc_train, "\nAUC Testing:", auc_test, "\n")

# Plot ROC (opsional)
roc_train_plot <- ggroc(roc_list_train)
roc_test_plot  <- ggroc(roc_list_test)
roc_train_plot + ggtitle(sprintf("ROC Decision Tree (Train AUC=%.3f, Test AUC=%.3f)", auc_train, auc_test))


#### Kesimpulan ----
cat("\n===== HASIL RINGKAS =====\n")
cat("Jumlah Train-Test:", nrow(training_set), "train -", nrow(test_set), "test\n")
cat("Akurasi (Train):", round(hasil_train["Akurasi"], 3), "\n")
cat("Akurasi (Test):", round(hasil_test["Akurasi"], 3), "\n")
cat("F1 (Train):", round(hasil_train["F1"], 3), "\n")
cat("F1 (Test):", round(hasil_test["F1"], 3), "\n")
cat("AUC (Train):", round(auc_train, 3), "\n")
cat("AUC (Test):", round(auc_test, 3), "\n")

if (hasil_train["Akurasi"] - hasil_test["Akurasi"] > 0.1) {
  cat("\nKesimpulan: Model cenderung OVERFITTING.\n")
} else if (hasil_test["Akurasi"] < 0.7) {
  cat("\nKesimpulan: Model cenderung UNDERFITTING.\n")
} else {
  cat("\nKesimpulan: Model FITTING dengan baik.\n")
}
