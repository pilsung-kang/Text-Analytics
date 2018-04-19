install.packages("tm")
install.packages("stringi")
install.packages("topicmodels")
install.packages("igraph")

library(tm)
library(stringi)
library(topicmodels)
library(igraph)

# Load the data
load("INS.RData")

# Construct the corpus for each journal with the abstracts
INS.Corpus <- Corpus(VectorSource(INS$Abstract))

# Preprocessing
# 1: to lower case
INS.Corpus <- tm_map(INS.Corpus, content_transformer(stri_trans_tolower))

# 2: remove puntuations
INS.Corpus <- tm_map(INS.Corpus, content_transformer(removePunctuation))

# 3. remove numbers
INS.Corpus <- tm_map(INS.Corpus, content_transformer(removeNumbers))

# 4. remove stopwords (en stopwords list)
myStopwords <- c(stopwords("en"))

INS.Corpus <- tm_map(INS.Corpus, removeWords, myStopwords)

# 5. Stemming
INS.Corpus <- tm_map(INS.Corpus, stemDocument)

# 6. remove stopwords (after stemming)
myStopwords <- c(stopwords("en"), "propos", "method", "algorithm", "elsevi", 
                 "result", "right","reserv", "paper", "present","studi")
INS.Corpus <- tm_map(INS.Corpus, removeWords, myStopwords)

# 7. Document-Term matrix
AbsDTM <- DocumentTermMatrix(INS.Corpus, control = list(minWordLength = 2))

# LDA with Gibbs Sampling
NTopic <- 30

ptm <- proc.time()
control_LDA_Gibbs <- list(alpha = 0.1, estimate.beta = TRUE, verbose = 0, prefix = tempfile(),
                          save = 0, keep = 0, seed = as.integer(Sys.time()), nstart = 1,
                          best = TRUE, delta = 0.1, iter = 5000, burnin = 0, thin = 2000)

Gibbs_LDA <-LDA(AbsDTM, NTopic, method = "Gibbs", control = control_LDA_Gibbs)
proc.time() - ptm
# LDA results (it took 240 seconds in my PC (i7-6770))

save.image(file = "INS_LDA_2017.RData")

load("INS_LDA_2017.RData")

# Initialize file names
TermFileName <- paste("AssignedTerms_", NTopic, ".csv", sep="")
CoTableFileName <- paste("CoTable_", NTopic, ".csv", sep="")
Top5.PaperFileName <- paste("Top5Papers_", NTopic, ".csv", sep="")

# Frequent terms
Gibbs.terms <- terms(Gibbs_LDA, 30)
write.csv(Gibbs.terms, TermFileName)

# Keyword co-occurence table for topics
Unique.terms <- rownames(table(as.character(Gibbs.terms)))
CoTable <- matrix(0, NTopic, NTopic)

for (a in 1:NTopic){
  for (b in 1:NTopic){
    CoTable[a,b] <- length(which(table(c(Gibbs.terms[,a],Gibbs.terms[,b])) == 2))    
  }
}

rownames(CoTable) <- paste("Topic", c(1:NTopic), sep="")
colnames(CoTable) <- paste("Topic", c(1:NTopic), sep="")

write.csv(CoTable, CoTableFileName)

# Topic Network
Topic.Graph <- graph.adjacency(CoTable, weighted=T, mode = "undirected")

# Remove loops
Topic.Graph <- simplify(Topic.Graph)

# Set labels and degrees of vertices
V(Topic.Graph)$label <- paste("Topic", c(1:30))
Topic.Graph <- delete.edges(Topic.Graph, which(E(Topic.Graph)$weight <= 3))
V(Topic.Graph)$degree <- degree(Topic.Graph)

# Plot the Graph
V(Topic.Graph)$label.cex <- 0.5*V(Topic.Graph)$degree/max(V(Topic.Graph)$degree)+1
V(Topic.Graph)$label.color <- rgb(0, 0, 0.2, 0.8)
V(Topic.Graph)$frame.color <- NA
egam1 <- 3*(log(E(Topic.Graph)$weight+1))/max(log(E(Topic.Graph)$weight+1))
E(Topic.Graph)$color <- rgb(0.5, 0.5, 0)
E(Topic.Graph)$width <- egam1

# plot the graph in layout
plot(Topic.Graph, layout=layout.kamada.kawai, vertex.size = 5, vertex.color = 8)

# Plot the communities
Topic.Community <- walktrap.community(Topic.Graph)
modularity(Topic.Community)
membership(Topic.Community)
plot(Topic.Community, Topic.Graph)


# Top 5 Papers in each Topic
Gibbs.topics <- topics(Gibbs_LDA, 1)
Topic.posterior <- posterior(Gibbs_LDA)$topics
Term.posterior <- posterior(Gibbs_LDA)$terms
Top5.Papers <- data.frame()

for (c in 1:NTopic){
  
  sel_idx <- which(Gibbs.topics == c)
  tmp_posterior <- data.frame(sel_idx, Topic.posterior[sel_idx, c])
  colnames(tmp_posterior) <- c("paper_idx", "posterior")
  tmp_posterior <- tmp_posterior[order(tmp_posterior$posterior, decreasing = TRUE),]
  tmp_topic <- paste("Topic_", c(c,c,c,c,c), sep="")
  tmp_papers <- cbind(tmp_topic, tmp_posterior[1:5,2], INS[tmp_posterior$paper_idx[1:5],])
  Top5.Papers <- rbind(Top5.Papers, tmp_papers)
}

colnames(Top5.Papers) <- c("Topic", "Posterior", "Title", "Year", "Abstract")

write.csv(Top5.Papers, Top5.PaperFileName)

# Hot & Cold Topics
HC.Data <- data.frame(INS$Year, Topic.posterior)
Year <- c(2007:2014)
Trend.Data <- data.frame()
  
for (i in 1:NTopic) {
  
  for (j in 1:length(Year)){
  
    Trend.Data[i,j] = mean(HC.Data[which(HC.Data[,1] == Year[j]),i+1])
  }
  
}

colnames(Trend.Data) <- c(2007:2014)
rownames(Trend.Data) <- paste("Topic", c(1:30), sep = "")

lr.slope <- data.frame()
  
for (i in 1:NTopic){
  
  lr.data <- as.data.frame(t(rbind(c(1:8), Trend.Data[i,])))
  colnames(lr.data) <- c("X", "Y")
  tmp.lr <- lm(Y ~ X, data = lr.data)
  lr.slope[i,1] <- i
  lr.slope[i,2] <- tmp.lr$coefficient[[2]]
}

lr.slope <- lr.slope[order(lr.slope[,2]),]

Hot5Topics <- lr.slope[NTopic:(NTopic-4),1]
Hot5Data <- t(Trend.Data[Hot5Topics,])

Cold5Topics <- lr.slope[1:5,1]
Cold5Data <- t(Trend.Data[Cold5Topics,])

par(mfcol = c(2,5))

for (i in 1:5){
  plot(c(2007:2014), Hot5Data[,i], type = "b", lty=1, pch = 15, col = rgb(1,0,0),
       main = colnames(Hot5Data)[i], xlab = "Year", ylab = "Mean theta", ylim = c(0, 0.20))
  plot(c(2007:2014), Cold5Data[,i], type = "b", lty=1, pch = 19, col = rgb(0,0,1),
       main = colnames(Cold5Data)[i], xlab = "Year", ylab = "Mean theta", ylim = c(0, 0.20)) 
}
dev.off()

