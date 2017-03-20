install.packages("tm", dependencies = TRUE)
install.packages("FSelector", dependencies = TRUE)
install.packages("stringi", dependencies = TRUE)

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
myStopwords <- c(stopwords("SMART"), "financ", "american", "associ", "firm",
                 "model", "data", "algorithm", "method", "imag", "ieee", "propos", 
                 "elsevi", "problem", "result", "find", "paper")

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

# Feature Selection 1: Information Gain
Journal.Binary.DTM <- data.frame(t(as.matrix(Journal.Binary.TDM)))
Journal.Binary.DTM <- data.frame(Journal = Journal.Data$Journal, Journal.Binary.DTM)

IG <- information.gain(Journal ~ ., Journal.Binary.DTM)
FS_IG <- data.frame(Terms, IG)
FS_IG <- FS_IG[order(FS_IG$attr_importance, decreasing = TRUE),]
sel.terms <- as.character(FS_IG$Terms[1:100])

sel.idx <- which(rownames(Journal.Frequency.TDM) %in% sel.terms == TRUE)
IG.Mat <- t(Journal.Frequency.TDM[sel.idx,])
rownames(IG.Mat) <- c(paste("TPAMI_", 1:433, sep=""), paste("JoF_", 1:521, sep=""))
IG.Mat <- as.data.frame(IG.Mat)

# Feature Selection 2: LSI
SVD.Mat <- svd(Journal.Frequency.TDM)
LSI.D <- SVD.Mat$d
LSI.U <- SVD.Mat$u
LSI.V <- SVD.Mat$v

# Select 2 features for documents
LSI.Mat <- as.data.frame(t((diag(100)*LSI.D[1:100]) %*% t(LSI.V[,1:100])))
rownames(LSI.Mat) <- c(paste("TPAMI_", 1:433, sep=""), paste("JoF_", 1:521, sep=""))
colnames(LSI.Mat) <- c(paste("SVD_", 1:100, sep=""))

save(IG.Mat, LSI.Mat, file = "FSJournal_2017.RData")
