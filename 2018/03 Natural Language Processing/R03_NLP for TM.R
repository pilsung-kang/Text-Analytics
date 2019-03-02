install.packages("tm")
install.packages("httr")
install.packages("XML")

library(tm)
library(httr)
library(XML)

data("crude")
crude[[1]]

MC_tokenizer(crude[[1]])
scan_tokenizer(crude[[1]])

# Stemming vs. Lemmatization
myCorpus <- Corpus(VectorSource(crude[[1]]$content))

# All characters to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))
# Remove punctuation
myCorpus <- tm_map(myCorpus, content_transformer(removePunctuation))
# Remove numbers
myCorpus <- tm_map(myCorpus, content_transformer(removeNumbers))

# Stemming
stemCorpus <- tm_map(myCorpus, stemDocument)

# Lemmatization
# Lemmatization function: require httr, XML packages
lemmatize <- function(wordlist) {
  get.lemma <- function(word, url) {
    response <- GET(url,query=list(spelling=word,standardize="",
                                   wordClass="",wordClass2="",
                                   corpusConfig="ncf",    # Nineteenth Century Fiction
                                   media="xml"))
    content <- content(response,type="text")
    xml     <- xmlInternalTreeParse(content)
    return(xmlValue(xml["//lemma"][[1]]))    
  }
  url <- "http://devadorner.northwestern.edu/maserver/lemmatizer"
  return(sapply(wordlist,get.lemma,url=url))
}

myTokens <- unlist(strsplit(as.character(myCorpus[[1]]), "\n"))
myTokens <- unlist(strsplit(myTokens, " "))
sel.idx <- which(nchar(myTokens) > 0)
myTokens <- myTokens[sel.idx]

LemmaCorpus <- lemmatize(myTokens)
LemmaCorpus <- paste(LemmaCorpus, collapse = " ")

# POS Tagging with MaxEnt
install.packages("openNLP")
library(openNLP)

s1 <- paste(c("Pierre Vinken, 61 years old, will join the board as a ",
             "nonexecutive director Nov. 29.\n",
             "Mr. Vinken is chairman of Elsevier N.V., ",
             "the Dutch publishing group."),
           collapse = "")
s1 <- as.String(s1)

## Need sentence and word token annotations.
s2 <- annotate(s1, list(Maxent_Sent_Token_Annotator(), Maxent_Word_Token_Annotator()))

## POS tag probabilities as (additional) features.
s3 <- annotate(s1, Maxent_POS_Tag_Annotator(probs = TRUE), s2)
