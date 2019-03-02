install.packages("tm", dependencies = TRUE)
install.packages("clValid", dependencies = TRUE)
install.packages("proxy", dependencies = TRUE)
install.packages("kohonen", dependencies = TRUE)

library(tm)
library(clValid)
library(proxy)
library(kohonen)

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
IG.Mat <- t(TFIDF)

Journal <- c(rep("TPAMI", 433), rep("JoF", 521))

# Clustering 1: K-Means Clustering ----------------------------------------
# Perform K-Means Clustering with K=5
KMC.IG <- kmeans(IG.Mat,5)
str(KMC.IG)

KMC.LSI <- kmeans(LSI.Mat,5)
str(KMC.LSI)

table(Journal, KMC.IG$cluster)
table(Journal, KMC.LSI$cluster)

# Clustering 2: Hierarchical Clustering -----------------------------------
# With Information Gain Features
IG.dist <- dist(IG.Mat, method = "cosine", diag = TRUE)

# Perform hierarchical clustering
IG.hr <- hclust(IG.dist, method = "average", members=NULL)
plot(IG.hr)

# Find the clusters
IG.hr.Clusters <- cutree(IG.hr, k=5)
table(Journal, IG.hr.Clusters)
rect.hclust(IG.hr, k=5, border="red")

source("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/My_R_Scripts/my.colorFct.R")
hr <- hclust(dist(IG.Mat, method = "cosine", diag = TRUE), method="complete")
hc <- hclust(dist(t(IG.Mat), method = "cosine", diag = TRUE), method="complete")
heatmap(IG.Mat, Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=my.colorFct(), scale="row")


# With LSI Features
LSI.dist <- dist(LSI.Mat, method = "cosine", diag = TRUE)

# Perform hierarchical clustering
LSI.hr <- hclust(LSI.dist, method = "average", members=NULL)
plot(LSI.hr)

# Find the clusters
LSI.hr.Clusters <- cutree(LSI.hr, k=5)
table(Journal, LSI.hr.Clusters)
rect.hclust(LSI.hr, k=5, border="red")

source("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/My_R_Scripts/my.colorFct.R")
hr <- hclust(dist(LSI.Mat, method = "cosine", diag = TRUE), method="complete")
hc <- hclust(dist(t(LSI.Mat), method = "cosine", diag = TRUE), method="complete")
heatmap(as.matrix(LSI.Mat), Rowv=as.dendrogram(hr), Colv=as.dendrogram(hc), col=my.colorFct(), scale="row")


# Clustering 3: SOM -----------------------------------------------------
# With IG Features
IG.som <- som(IG.Mat, grid = somgrid(5,5,"hexagonal"))
str(IG.som)
IG.Map <- map(IG.som, IG.Mat)

plot(IG.som, type = "quality")
plot(IG.som, type = "codes")
plot(IG.som, type = "counts")
plot(IG.som, type = "dist.neighbours")
plot(IG.som, type = "mapping", labels = rownames(IG.Mat), col = c(rep(4,100), rep(2,100)))
som.hc <- cutree(hclust(dist(IG.som$codes[[1]], method = "cosine"), method = "average"), 5)
add.cluster.boundaries(IG.som, som.hc)

# With LSI Features
LSI.som <- som(as.matrix(LSI.Mat), grid = somgrid(5,5,"hexagonal"))
str(LSI.som)
LSI.Map <- map(LSI.som, as.matrix(LSI.Mat))

plot(LSI.som, type = "quality")
plot(LSI.som, type = "codes")
plot(LSI.som, type = "counts")
plot(LSI.som, type = "dist.neighbours")
plot(LSI.som, type = "mapping", labels = rownames(IG.Mat), col = c(rep(4,100), rep(2,100)))
som.hc <- cutree(hclust(dist(LSI.som$codes[[1]], method = "cosine"), method = "average"), 5)
add.cluster.boundaries(LSI.som, som.hc)

