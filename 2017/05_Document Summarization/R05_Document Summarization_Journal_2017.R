install.packages("tm", dependencies = TRUE)
install.packages("wordcloud", dependencies = TRUE)
install.packages("arules", dependencies = TRUE)
install.packages("arulesViz", dependencies = TRUE)
install.packages("igraph", dependencies = TRUE)
install.packages("stringi", dependencies = TRUE)

library(tm)
library(wordcloud)
library(arules)
library(arulesViz)
library(igraph)
library(stringi)

# Load the data
TPAMI <- read.csv("IEEE_TPAMI_2015_2017.csv", encoding = "UTF-8", stringsAsFactors = FALSE)
JoF <- read.csv("Journal of Banking and Finance_2015_2017.csv", encoding = "UTF-8", stringsAsFactors = FALSE)

# Construct the corpus for each journal with the abstracts
TPAMI.Corpus <- Corpus(VectorSource(TPAMI$Abstract))
JoF.Corpus <- Corpus(VectorSource(JoF$Abstract))

# Preprocessing
# 1: to lower case
TPAMI.Corpus <- tm_map(TPAMI.Corpus, content_transformer(stri_trans_tolower))
JoF.Corpus <- tm_map(JoF.Corpus, content_transformer(stri_trans_tolower))

# 2: remove puntuations
TPAMI.Corpus <- tm_map(TPAMI.Corpus, content_transformer(removePunctuation))
JoF.Corpus <- tm_map(JoF.Corpus, content_transformer(removePunctuation))

# 3. remove numbers
TPAMI.Corpus <- tm_map(TPAMI.Corpus, content_transformer(removeNumbers))
JoF.Corpus <- tm_map(JoF.Corpus, content_transformer(removeNumbers))

# 4. remove stopwords (SMART stopwords list)
myStopwords <- c(stopwords("SMART"))

TPAMI.Corpus <- tm_map(TPAMI.Corpus, removeWords, myStopwords)
JoF.Corpus <- tm_map(JoF.Corpus, removeWords, myStopwords)

# 5. Stemming
TPAMI.Corpus <- tm_map(TPAMI.Corpus, stemDocument)
JoF.Corpus <- tm_map(JoF.Corpus, stemDocument)

# 4. remove stopwords (with frequently used words)
myStopwords <- c(stopwords("SMART"), "financ", "american", "associ", "firm",
                 "model", "data", "algorithm", "method", "imag", "ieee", "propos", 
                 "elsevi", "problem", "result", "find", "paper")

TPAMI.Corpus <- tm_map(TPAMI.Corpus, removeWords, myStopwords)
JoF.Corpus <- tm_map(JoF.Corpus, removeWords, myStopwords)

# Term-Document Matrix
TPAMI.TDM <- TermDocumentMatrix(TPAMI.Corpus, control = list(minWordLength = 1))
JoF.TDM <- TermDocumentMatrix(JoF.Corpus, control = list(minWordLength = 1))

as.matrix(TPAMI.TDM)[11:30,11:30]
as.matrix(JoF.TDM)[11:30,11:30]

# Frequently used words
findFreqTerms(TPAMI.TDM, lowfreq=100)
findFreqTerms(JoF.TDM, lowfreq=100)

# Construct a Word Cloud with IEEE_TPAMI abstracts
TPAMI.wcmat <- as.matrix(TPAMI.TDM)

# calculate the frequency of words
TPAMI.word.freq <- sort(rowSums(TPAMI.wcmat), decreasing=TRUE)
TPAMI.keywords <- names(TPAMI.word.freq)
TPAMI.wcdat <- data.frame(word = TPAMI.keywords, freq = TPAMI.word.freq)

pal <- brewer.pal(8, "Dark2")
wordcloud(TPAMI.wcdat$word, TPAMI.wcdat$freq, min.freq=10, scale = c(5, 0.2), 
          rot.per = 0.1, col=pal, random.order=F)

# Construct a Word Cloud with JoF abstracts
JoF.wcmat <- as.matrix(JoF.TDM)

# calculate the frequency of words
JoF.word.freq <- sort(rowSums(JoF.wcmat), decreasing=TRUE)
JoF.keywords <- names(JoF.word.freq)
JoF.wcdat <- data.frame(word = JoF.keywords, freq = JoF.word.freq)

pal <- brewer.pal(8, "Dark2")
wordcloud(JoF.wcdat$word, JoF.wcdat$freq, min.freq=10, scale = c(5, 0.2), 
          rot.per = 0.1, col=pal, random.order=F)

# Association Rules for IEEE_TPAMI
TPAMI.tran <- as.matrix(t(TPAMI.TDM))
TPAMI.tran <- as(TPAMI.tran, "transactions")

TPAMI.rules <- apriori(TPAMI.tran, parameter=list(minlen=2,supp=0.04, conf=0.75))
inspect(TPAMI.rules)

TPAMI.rules.sorted <- sort(TPAMI.rules, by="lift")
subset.matrix <- is.subset(TPAMI.rules.sorted, TPAMI.rules.sorted)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
TPAMI.rules.pruned <- TPAMI.rules.sorted[!redundant]
inspect(TPAMI.rules.pruned)

# Plot the rules
plot(TPAMI.rules.pruned, method="graph")

# Association Rules for Journal of Banking and Fiance
JoF.tran <- as.matrix(t(JoF.TDM))
JoF.tran <- as(JoF.tran, "transactions")

JoF.rules <- apriori(JoF.tran, parameter=list(minlen=2, supp=0.04, conf=0.7))
inspect(JoF.rules)

JoF.rules.sorted <- sort(JoF.rules, by="lift")
subset.matrix <- is.subset(JoF.rules.sorted, JoF.rules.sorted)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
JoF.rules.pruned <- JoF.rules.sorted[!redundant]
inspect(JoF.rules.pruned)

# Plot the rules
plot(JoF.rules.pruned, method="graph")

# Section 3: Keyword Network ----------------------------------------------
# For IEEE_TPAMI Abstracts
# Change it to a Boolean matrix
TPAMI.wcmat[TPAMI.wcmat >= 1] <- 1
# find the words that are used more than 30 times
freq.idx <- which(rowSums(TPAMI.wcmat) >= 30)
TPAMI.wcmat.freq <- TPAMI.wcmat[freq.idx,]

# Transform into a term-term adjacency matrix
TPAMI.ttmat <- TPAMI.wcmat.freq %*% t(TPAMI.wcmat.freq)

# inspect terms numbered 1 to 10
TPAMI.ttmat[1:10,1:10]

# Build a graph from the above matrix
TPAMI.graph <- graph.adjacency(TPAMI.ttmat, weighted=T, mode = "undirected")

# remove loops
TPAMI.graph <- simplify(TPAMI.graph)

# set labels and degrees of vertices
V(TPAMI.graph)$label <- V(TPAMI.graph)$name
TPAMI.graph <- delete.edges(TPAMI.graph, which(E(TPAMI.graph)$weight <= 30))

TPAMI.graph <- delete.vertices(TPAMI.graph, which(degree(TPAMI.graph) == 0))
V(TPAMI.graph)$degree <- degree(TPAMI.graph)

# set seed to make the layout reproducible
set.seed(3952)
plot(TPAMI.graph, layout=layout.fruchterman.reingold)
plot(TPAMI.graph, layout=layout.kamada.kawai, 
     vertex.size = 10, vertex.color = 4, vertex.label.cex = 1)

# Make the network look better
V(TPAMI.graph)$label.cex <- 0.5*V(TPAMI.graph)$degree/max(V(TPAMI.graph)$degree)+1
V(TPAMI.graph)$label.color <- rgb(0, 0, 0.2, 0.8)
V(TPAMI.graph)$frame.color <- NA
egam <- 3*(log(E(TPAMI.graph)$weight+1))/max(log(E(TPAMI.graph)$weight+1))
E(TPAMI.graph)$color <- rgb(0.8, 0.5, 0)
E(TPAMI.graph)$width <- egam

# plot the graph in layout
plot(TPAMI.graph, layout=layout.kamada.kawai, vertex.size = 10, vertex.color = 7)

# Plot the communities
TPAMI.community <- walktrap.community(TPAMI.graph)
modularity(TPAMI.community)
membership(TPAMI.community)
plot(TPAMI.community, TPAMI.graph)

# For Journal of Banking and Finance Abstracts
# Change it to a Boolean matrix
JoF.wcmat[JoF.wcmat >= 1] <- 1
# find the words that are used more than 10 times
freq.idx <- which(rowSums(JoF.wcmat) >= 30)
JoF.wcmat.freq <- JoF.wcmat[freq.idx,]

# Transform into a term-term adjacency matrix
JoF.ttmat <- JoF.wcmat.freq %*% t(JoF.wcmat.freq)

# inspect terms numbered 5 to 10
JoF.ttmat[1:10,1:10]

# Build a graph from the above matrix
JoF.graph <- graph.adjacency(JoF.ttmat, weighted=T, mode = "undirected")

# remove loops
JoF.graph <- simplify(JoF.graph)

# set labels and degrees of vertices
V(JoF.graph)$label <- V(JoF.graph)$name
JoF.graph <- delete.edges(JoF.graph, which(E(JoF.graph)$weight <= 20))

JoF.graph <- delete.vertices(JoF.graph, which(degree(JoF.graph) == 0))
V(JoF.graph)$degree <- degree(JoF.graph)

# set seed to make the layout reproducible
set.seed(3952)
plot(JoF.graph, layout=layout.fruchterman.reingold)
plot(JoF.graph, layout=layout.kamada.kawai, 
     vertex.size = 13, vertex.color = 4, vertex.label.cex = 1)

# Make the network look better
V(JoF.graph)$label.cex <- 0.5*V(JoF.graph)$degree/max(V(JoF.graph)$degree)+1
V(JoF.graph)$label.color <- rgb(0, 0, 0.2, 0.8)
V(JoF.graph)$frame.color <- NA
egam <- 3*(log(E(JoF.graph)$weight+1))/max(log(E(JoF.graph)$weight+1))
E(JoF.graph)$color <- rgb(0.8, 0.5, 0)
E(JoF.graph)$width <- egam

# plot the graph in layout1
plot(JoF.graph, layout=layout.kamada.kawai, vertex.size = 10, vertex.color = 7)

# Plot the communities
JoF.community <- walktrap.community(JoF.graph)
modularity(JoF.community)
membership(JoF.community)
plot(JoF.community, JoF.graph)
