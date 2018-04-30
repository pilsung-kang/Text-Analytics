# Performance Evaluation Function -----------------------------------------
perf_eval <- function(cm){
  
  # Recall
  Recall = cm[2,2]/sum(cm[2,])
  # Precision
  Precision = cm[2,2]/sum(cm[,2])
  # Accuracy
  Acc = (cm[1,1]+cm[2,2])/sum(cm)
  # F1
  F1 = 2*Recall*Precision/(Recall+Precision)
  
  return(c(Recall, Precision, Acc, F1))
}

# Load the data
load("FSJournal_2017.RData")

# For IG matrix: log(tf)*idf
log_tf <- log(1+t(IG.Mat))

# (Inverse) Document Frequency
N_total <- dim(log_tf)[2]
n_doc <- function(x) length(which(x > 0))
idf_vec <- log(N_total/apply(log_tf, 1, n_doc))
idf_mat <- replicate(N_total, idf_vec)

# TF-IDF
TFIDF <- log_tf*idf_mat
IG.Mat <- as.data.frame(t(TFIDF))

Journal <- c(rep("TPAMI", 433), rep("JoF", 521))

# Classification data sets
IG.Data <- data.frame(Journal, IG.Mat)
LSI.Data <- data.frame(Journal, LSI.Mat)

# Training/Validation
trn.idx <- c(1:333, 434:854)
val.idx <- c(334:433, 855:954)

IG.Data.Trn <- IG.Data[trn.idx,]
IG.Data.Val <- IG.Data[val.idx,]

LSI.Data.Trn <- LSI.Data[trn.idx,]
LSI.Data.Val <- LSI.Data[val.idx,]

Class.Perf.IG <- matrix(0, 4, 4)
rownames(Class.Perf.IG) <- c("Naive Bayes", "k-NN", "CTree", "SVM")
colnames(Class.Perf.IG) <- c("Recall", "Precision", "Accuracy", "F1-measure")

Class.Perf.LSI <- matrix(0, 4, 4)
rownames(Class.Perf.LSI) <- c("Naive Bayes", "k-NN", "CTree", "SVM")
colnames(Class.Perf.LSI) <- c("Recall", "Precision", "Accuracy", "F1-measure")

# Classification 1: Naive Bayesian Classifier ----------------------------------------
# e1071 package install & call
install.packages("e1071", dependencies = TRUE)
library(e1071)

# Selected Features based on Information Gain
NB.IG <- naiveBayes(Journal ~ ., data = IG.Data.Trn)
NB.IG$apriori
NB.IG$tables

# Predict the new input data based on Naive Bayesian Classifier
NB.IG.Posterior = predict(NB.IG, IG.Data.Val, type = "raw")
NB.IG.PreY = predict(NB.IG, IG.Data.Val, type ="class")

# Confusion Matrix & Performance Evaluation
CF.NB.IG <- table(IG.Data.Val$Journal, NB.IG.PreY)
Class.Perf.IG[1,] <- perf_eval(CF.NB.IG)
Class.Perf.IG

# Extracted Features based on LSI
NB.LSI <- naiveBayes(Journal ~ ., data = LSI.Data.Trn)
NB.LSI$apriori
NB.LSI$tables

# Predict the new input data based on Naive Bayesian Classifier
NB.LSI.Posterior = predict(NB.LSI, LSI.Data.Val, type = "raw")
NB.LSI.PreY = predict(NB.LSI, LSI.Data.Val, type ="class")

# Confusion Matrix & Performance Evaluation
CF.NB.LSI <- table(LSI.Data.Val$Journal, NB.LSI.PreY)
Class.Perf.LSI[1,] <- perf_eval(CF.NB.LSI)
Class.Perf.LSI

# Classification 2: k-Nearest Neighbor Classifier ----------------------------------------
# kknn package install & call
install.packages("kknn", dependencies = TRUE)
library(kknn)

# Normalize the input data
IG.Data.Scaled <- data.frame(Journal, scale(IG.Mat, center = TRUE, scale = TRUE))
LSI.Data.Scaled <- data.frame(Journal, scale(LSI.Mat, center = TRUE, scale = TRUE))

IG.Data.Scaled.Trn <- IG.Data.Scaled[trn.idx,]
IG.Data.Scaled.Val <- IG.Data.Scaled[val.idx,]

LSI.Data.Scaled.Trn <- LSI.Data.Scaled[trn.idx,]
LSI.Data.Scaled.Val <- LSI.Data.Scaled[val.idx,]

# Leave-one-out validation for finding the best k: Information Gain Features
IG.knn <- train.kknn(Journal ~ ., IG.Data.Scaled.Trn, kmax = 10, distance = 2)
IG.knn$MISCLASS
IG.knn$best.parameters

# Tranining the k-nn model with the best parameters
IG.knn <- kknn(Journal ~ ., IG.Data.Scaled.Trn, IG.Data.Scaled.Val, k = IG.knn$best.parameters$k, 
               distance = 2, kernel = IG.knn$best.parameters$kernel)

# Prediction
IG.knn.PreY <- fitted(IG.knn)

# Confusion Matrix & Performance Evaluation
CF.knn.IG <- table(IG.Data.Val$Journal, IG.knn.PreY)
Class.Perf.IG[2,] <- perf_eval(CF.knn.IG)
Class.Perf.IG

# Leave-one-out validation for finding the best k: LSI Features
LSI.knn <- train.kknn(Journal ~ ., LSI.Data.Scaled.Trn, kmax = 10, distance = 2)
LSI.knn$MISCLASS
LSI.knn$best.parameters

# Tranining the k-nn model with the best parameters
LSI.knn <- kknn(Journal ~ ., LSI.Data.Scaled.Trn, LSI.Data.Scaled.Val, k = LSI.knn$best.parameters$k, 
               distance = 2, kernel = LSI.knn$best.parameters$kernel)

# Prediction
LSI.knn.PreY <- fitted(LSI.knn)

# Confusion Matrix & Performance Evaluation
CF.knn.LSI <- table(LSI.Data.Val$Journal, LSI.knn.PreY)
Class.Perf.LSI[2,] <- perf_eval(CF.knn.LSI)
Class.Perf.LSI

# Classification 3: Classification Tree ----------------------------------------
# party package install & call
install.packages("party")
library(party)

# A Simple validation for finding the best tree parameters: Information Gain Features
# tree parameter settings
Min.Criterion = c(0.7, 0.8, 0.9)
Min.Split = c(3, 5, 10)
Max.Depth = c(0, 5, 10)
IG.LOO.Result = matrix(0,length(Min.Criterion)*length(Min.Split)*length(Max.Depth),8)

iter.cnt = 1

for (i in 1:length(Min.Criterion))
{
  for ( j in 1:length(Min.Split))
  {
    for ( k in 1:length(Max.Depth))
    {
            
      cat("CART Min criterion:", Min.Criterion[i], ", Min split:", Min.Split[j], ", Max depth:", Max.Depth[k], "\n")
      Tmp.Control = ctree_control(mincriterion = Min.Criterion[i], minsplit = Min.Split[j], maxdepth = Max.Depth[k])
      
      Tmp.Trn <- IG.Data.Trn[c(1:233, 334:654),]
      Tmp.Val <- IG.Data.Trn[c(234:333, 655:754),]
        
      Tmp.Tree <- ctree(Journal ~ ., data = Tmp.Trn, controls = Tmp.Control)
      Tmp.PreY <- predict(Tmp.Tree, newdata = Tmp.Val)
      
      IG.LOO.Result[iter.cnt,1] <- Min.Criterion[i]
      IG.LOO.Result[iter.cnt,2] <- Min.Split[j]
      IG.LOO.Result[iter.cnt,3] <- Max.Depth[k]
      IG.LOO.Result[iter.cnt,4:7] <- perf_eval(table(Tmp.Val$Journal, Tmp.PreY))
      IG.LOO.Result[iter.cnt,8] <- length(nodes(Tmp.Tree, unique(where(Tmp.Tree))))
      iter.cnt = iter.cnt + 1
    }    
  }
}

IG.LOO.Result <- IG.LOO.Result[order(IG.LOO.Result[,7], decreasing = T),]
IG.LOO.Result

Best.IG.Criterion <- IG.LOO.Result[1,1]
Best.IG.Split <- IG.LOO.Result[1,2]
Best.IG.Depth <- IG.LOO.Result[1,3]

IG.Best.Control = ctree_control(mincriterion = Best.IG.Criterion, minsplit = Best.IG.Split, maxdepth = Best.IG.Depth)

IG.Tree <- ctree(Journal ~ ., data = IG.Data.Trn, controls = IG.Best.Control)
IG.Tree.PreY <- predict(IG.Tree, newdata = IG.Data.Val)

CF.Tree.IG <- table(IG.Data.Val$Journal, IG.Tree.PreY)
Class.Perf.IG[3,] <- perf_eval(CF.Tree.IG)
Class.Perf.IG

plot(IG.Tree)

# A Simple validation for finding the best tree parameters: LSI Features
# tree parameter settings
Min.Criterion = c(0.7, 0.8, 0.9)
Min.Split = c(3, 5, 10)
Max.Depth = c(0, 5, 10)
LSI.LOO.Result = matrix(0,length(Min.Criterion)*length(Min.Split)*length(Max.Depth),8)

iter.cnt = 1

for (i in 1:length(Min.Criterion))
{
  for ( j in 1:length(Min.Split))
  {
    for ( k in 1:length(Max.Depth))
    {
      
      cat("CART Min criterion:", Min.Criterion[i], ", Min split:", Min.Split[j], ", Max depth:", Max.Depth[k], "\n")
      Tmp.Control = ctree_control(mincriterion = Min.Criterion[i], minsplit = Min.Split[j], maxdepth = Max.Depth[k])
      
      Tmp.Trn <- LSI.Data.Trn[c(1:233, 334:654),]
      Tmp.Val <- LSI.Data.Trn[c(234:333, 655:754),]
      
      Tmp.Tree <- ctree(Journal ~ ., data = Tmp.Trn, controls = Tmp.Control)
      Tmp.PreY <- predict(Tmp.Tree, newdata = Tmp.Val)
      
      LSI.LOO.Result[iter.cnt,1] <- Min.Criterion[i]
      LSI.LOO.Result[iter.cnt,2] <- Min.Split[j]
      LSI.LOO.Result[iter.cnt,3] <- Max.Depth[k]
      LSI.LOO.Result[iter.cnt,4:7] <- perf_eval(table(Tmp.Val$Journal, Tmp.PreY))
      LSI.LOO.Result[iter.cnt,8] <- length(nodes(Tmp.Tree, unique(where(Tmp.Tree))))
      iter.cnt = iter.cnt + 1
    }    
  }
}

LSI.LOO.Result <- LSI.LOO.Result[order(LSI.LOO.Result[,7], decreasing = T),]
LSI.LOO.Result

Best.LSI.Criterion <- LSI.LOO.Result[1,1]
Best.LSI.Split <- LSI.LOO.Result[1,2]
Best.LSI.Depth <- LSI.LOO.Result[1,3]

LSI.Best.Control = ctree_control(mincriterion = Best.LSI.Criterion, minsplit = Best.LSI.Split, maxdepth = Best.LSI.Depth)

LSI.Tree <- ctree(Journal ~ ., data = LSI.Data.Trn, controls = LSI.Best.Control)
LSI.Tree.PreY <- predict(LSI.Tree, newdata = LSI.Data.Val)

CF.Tree.LSI <- table(LSI.Data.Val$Journal, LSI.Tree.PreY)
Class.Perf.LSI[3,] <- perf_eval(CF.Tree.LSI)
Class.Perf.LSI

plot(LSI.Tree)

# Classification 4: Support Vector Machine ----------------------------------------
# party package install & call
install.packages("e1071")
library(e1071)

# Parameter Search: Information Gain Features
IG.Param.Search <- tune(svm, Journal ~ ., data = IG.Data.Scaled.Trn, 
                        ranges = list(gamma = 2^(-10:0), cost = 2^(-3:5)), tunecontrol = tune.control(sampling = "fix"))

IG.Param.Search

Best.Gamma <- IG.Param.Search$best.parameters$gamma
Best.Cost <- IG.Param.Search$best.parameters$cost

# Train the SVM and prediction
IG.SVM <- svm(Journal ~ ., data = IG.Data.Scaled.Trn, gamma = Best.Gamma, cost = Best.Cost)
IG.SVM.PreY <- predict(IG.SVM, newdata = IG.Data.Val)

CF.SVM.IG <- table(IG.Data.Val$Journal, IG.SVM.PreY)
Class.Perf.IG[4,] <- perf_eval(CF.SVM.IG)
Class.Perf.IG

# Parameter Search: LSI Features
LSI.Param.Search <- tune(svm, Journal ~ ., data = LSI.Data.Scaled.Trn, 
                        ranges = list(gamma = 2^(-10:0), cost = 2^(-3:5)), tunecontrol = tune.control(sampling = "fix"))

LSI.Param.Search

Best.Gamma <- LSI.Param.Search$best.parameters$gamma
Best.Cost <- LSI.Param.Search$best.parameters$cost

# Train the SVM and prediction
LSI.SVM <- svm(Journal ~ ., data = LSI.Data.Scaled.Trn, gamma = Best.Gamma, cost = Best.Cost)
LSI.SVM.PreY <- predict(LSI.SVM, newdata = LSI.Data.Val)

CF.SVM.LSI <- table(LSI.Data.Val$Journal, LSI.SVM.PreY)
Class.Perf.LSI[4,] <- perf_eval(CF.SVM.LSI)
Class.Perf.LSI

# Summary Result
Class.Perf.IG
Class.Perf.LSI
