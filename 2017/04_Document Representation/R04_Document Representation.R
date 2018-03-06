install.packages("RWeka", dependencies = TRUE)
install.packages("tm", dependencies = TRUE)

library(RWeka)
library(tm)

load("SpeechData.RData")

# Construct corpuses
# VectorSource specifies that the source is character vectors.
myCorpus <- Corpus(VectorSource(SpeechData$Speech))

# Preprocessing
# 1: to lower case
PreCorpus <- tm_map(myCorpus, content_transformer(tolower))

# 2: remove puntuations
PreCorpus <- tm_map(PreCorpus, content_transformer(removePunctuation))

# 3. remove numbers
PreCorpus <- tm_map(PreCorpus, content_transformer(removeNumbers))

# 4. remove stopwords (SMART stopwords list + obama + romney)
myStopwords <- c(stopwords("SMART"), "obama", "romney")

PreCorpus <- tm_map(PreCorpus, removeWords, myStopwords)

# 5. Stemming
stemCorpus <- tm_map(PreCorpus, stemDocument)

# Term-Document Matrix: Without Preprocessing
myTDM <- TermDocumentMatrix(myCorpus, control = list(minWordLength = 1))

# Term-Document Matrix: With Preprocessing
stemTDM <- TermDocumentMatrix(stemCorpus, control = list(minWordLength = 1))

# Check the term-document matrix
as.matrix(stemTDM)[1:30,1:20]

# TF-IDF Variants
# TF 1: Natural
natural_tf <- as.matrix(stemTDM)

# TF 2: Logarithm
log_tf <- log(1+natural_tf)

# TF 3: augmented
max_tf <- apply(natural_tf, 2, max)
augmented_tf <- 0.5+0.5*t(t(natural_tf)/max_tf)

# (Inverse) Document Frequency
N_total <- dim(natural_tf)[2]

n_doc <- function(x) length(which(x > 0))

idf_vec <- log(N_total/apply(natural_tf, 1, n_doc))
idf_mat <- replicate(N_total, idf_vec)

# TF-IDF 1: Natrual TF * IDF
TFIDF_1 <- natural_tf*idf_mat

# TF-IDF 2: Log TF * IDF (cosine normalized)
TFIDF_2 <- log_tf*idf_mat

cos_normalize <- function(x) x/sqrt(sum(x^2))

TFIDF_2 <- apply(TFIDF_2, 2, cos_normalize)

# Construct N-gram matrix
tmpTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 3))
NgramTDM <- TermDocumentMatrix(stemCorpus, control = list(tokenize = tmpTokenizer))
inspect(NgramTDM[30000:30100,1:20])