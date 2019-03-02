install.packages("tm")
install.packages("FSelector")
install.packages("stringi")

library(tm)
library(FSelector)
library(stringi)

# Load the data
TPAMI <- read.csv("IEEE_TPAMI_2015_2017.csv", encoding = "UTF-8", stringsAsFactors = FALSE)
JoF <- read.csv("Journal of Banking and Finance_2015_2017.csv", encoding = "UTF-8", stringsAsFactors = FALSE)

Journal.Data <- rbind(TPAMI[,c(4,12)], JoF[,c(4,12)])
names(Journal.Data) <- c("Journal", "Abstract")

# Construct the corpus for each journal with the abstracts
Journal.Corpus <- Corpus(VectorSource(Journal.Data$Abstract))

# Preprocessing
# 1: to lower case
Journal.Corpus <- tm_map(Journal.Corpus, content_transformer(stri_trans_tolower))

# 2: remove puntuations
Journal.Corpus <- tm_map(Journal.Corpus, content_transformer(removePunctuation))

# 3. remove numbers
Journal.Corpus <- tm_map(Journal.Corpus, content_transformer(removeNumbers))

# 4. remove stopwords (SMART stopwords list)
myStopwords <- c(stopwords("SMART"))

Journal.Corpus <- tm_map(Journal.Corpus, removeWords, myStopwords)

# 5. Stemming
Journal.Corpus <- tm_map(Journal.Corpus, stemDocument)

# Term-Document Matrix
Journal.TDM <- TermDocumentMatrix(Journal.Corpus, control = list(minWordLength = 1))

Journal.Frequency.TDM <- as.matrix(Journal.TDM)

pos_idx <- which(Journal.Frequency.TDM >= 1)
Journal.Binary.TDM <- Journal.Frequency.TDM
Journal.Binary.TDM[pos_idx] <- 1
Terms <- rownames(Journal.Binary.TDM)

# Remove multibyte strings
Terms <- iconv(Terms, "latin1", "ASCII", sub="")
rownames(Journal.Frequency.TDM) <- Terms
rownames(Journal.Binary.TDM) <- Terms

# Dimensionality Reduction 1: Feature Selection ---------------------------
Journal.Binary.DTM <- data.frame(t(as.matrix(Journal.Binary.TDM)))
Journal.Binary.DTM <- data.frame(Journal = Journal.Data$Journal, Journal.Binary.DTM)

# Metric 1: Acc
pos_idx <- which(Journal.Binary.DTM$Journal == "IEEE Transactions on Pattern Analysis and Machine Intelligence")
Acc <- sapply(Journal.Binary.DTM[,-1], function(x) sum(x[pos_idx])-sum(x[-pos_idx]))
FS_Acc <- data.frame(Terms, Acc)
# Sort the terms w.r.t. Acc
FS_Acc <- FS_Acc[order(FS_Acc$Acc, decreasing = TRUE),]
FS_Acc[1:30,]
FS_Acc[(length(Terms)-30):length(Terms),]

# Metric 2: F1-Measure
F1 <- sapply(Journal.Binary.DTM[,-1], function(x) 2*sum(x[pos_idx])/(length(pos_idx)+sum(x)))
FS_F1 <- data.frame(Terms, F1)
# Sort the terms w.r.t. F1
FS_F1 <- FS_F1[order(FS_F1$F1, decreasing = TRUE),]
FS_F1[1:30,]
FS_F1[(length(Terms)-30):length(Terms),]

# Metric 3: Chi-squared
CS <- chi.squared(Journal ~ ., Journal.Binary.DTM)
FS_CS <- data.frame(Terms, CS)
FS_CS <- FS_CS[order(FS_CS$attr_importance, decreasing = TRUE),]
FS_CS[1:30,]
FS_CS[(length(Terms)-30):length(Terms),]

# Metric 4: Information Gain
IG <- information.gain(Journal ~ ., Journal.Binary.DTM)
FS_IG <- data.frame(Terms, IG)
FS_IG <- FS_IG[order(FS_IG$attr_importance, decreasing = TRUE),]
FS_IG[1:30,]
FS_IG[(length(Terms)-30):length(Terms),]

# Combine Top 30 significant terms for the positivie class (TMAPI)
Top30.Terms <- data.frame(FS_Acc[1:30,], FS_F1[1:30,], FS_CS[1:30,], FS_IG[1:30,])
names(Top30.Terms) <- c("Terms_Acc", "Acc_Score", "Terms_F1", "F1_Score", 
                        "Terms_Chi-Squared", "Chi-Squared Score", "Terms_Information Gain", "Information Gain_Score")

write.csv(Top30.Terms, file = "Top30 Terms for TPAMI.csv")

# Dimensionality Reduction 2-1: PCA ---------------------------------------
Journal.Frequency.DTM <- data.frame(t(as.matrix(Journal.Frequency.TDM)))
PCs <- prcomp(Journal.Frequency.DTM, scale = TRUE)
summary(PCs)
screeplot(PCs, npcs = 100, type = "lines", main = "Variance explained by each principal component")

Journal.PCs <- predict(PCs)

# Plot the articles in 2-dim
par(mfrow = c(1,2))
plot(Journal.PCs[,1:2], type = "n", xlim = c(-10,10), ylim = c(-5,5))
text(Journal.PCs[1:30,1], Journal.PCs[1:30,2], label = paste("TPAMI_", 1:30, sep = ""), col=4)
text(Journal.PCs[434:463,1], Journal.PCs[434:463,2], label = paste("JoF_", 1:30, sep = ""), col=10)

plot(Journal.PCs[,3:4], type = "n", xlim = c(-5,5), ylim = c(-2,5))
text(Journal.PCs[1:30,3], Journal.PCs[1:30,4], label = paste("TPAMI_", 1:30, sep = ""), col=4)
text(Journal.PCs[434:463,3], Journal.PCs[434:463,4], label = paste("JoF_", 1:30, sep = ""), col=10)


# Dimensionality Reduction 2-2: MDS ---------------------------------------
Journal.Cor <- cor(Journal.Frequency.TDM)
Journal.Dist <- 1-Journal.Cor
Journal.Dist <- as.data.frame(Journal.Dist)
    
MDS <- cmdscale(Journal.Dist, eig = TRUE, k = 10)
Journal.MDS <- MDS$points

par(mfrow = c(1,1))
plot(Journal.MDS[,1], Journal.MDS[,2], xlab = "Coordinate 1", ylab = "Coordinate 2", 
     main = "MDS plot", type = "n")
text(Journal.MDS[1:30,1], Journal.MDS[1:30,2], label = paste("TPAMI_", 1:30, sep = ""), col=4)
text(Journal.MDS[434:463,1], Journal.MDS[434:463,2], label = paste("JoF_", 1:30, sep = ""), col=10)


# Dimensionality Reduction 2-3: LSI ---------------------------------------
SVD.Mat <- svd(Journal.Frequency.TDM)
LSI.D <- SVD.Mat$d
LSI.U <- SVD.Mat$u
LSI.V <- SVD.Mat$v

# Plot the singluar vectors
plot(1:length(LSI.D), LSI.D)

# Select 2 features for documents
Document.Mat <- t((diag(2)*LSI.D[1:2]) %*% t(LSI.V[,1:2]))

plot(Document.Mat[,1], Document.Mat[,2], xlab = "SVD 1", ylab = "SVD 2", 
     main = "LSI plot for Documents", type = "n")
text(Document.Mat[1:30,1], Document.Mat[1:30,2], label = paste("TPAMI_", 1:30, sep = ""), col=4)
text(Document.Mat[434:463,1], Document.Mat[434:463,2], label = paste("JoF_", 1:30, sep = ""), col=10)

# Select 2 features for Terms
Term.Mat <- LSI.U[,1:2]  %*% (diag(2)*LSI.D[1:2])

plot(Term.Mat[,1], Term.Mat[,2], xlab = "SVD 1", ylab = "SVD 2", type = "n", 
     main = "LSI plot for Terms")
text(Term.Mat[,1], Term.Mat[,2], label = Terms)

plot(Term.Mat[,1], Term.Mat[,2], xlab = "SVD 1", ylab = "SVD 2", type = "n", 
     xlim = c(0.04,0.06), ylim = c(0.04,0.06), main = "LSI plot for Terms")
text(Term.Mat[,1], Term.Mat[,2], label = Terms)
