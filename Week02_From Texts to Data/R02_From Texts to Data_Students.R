# Case 1-1: Import data from a file ---------------------------------------

# If sentences are separated
speech1 <- readLines("Inaugural Address Lines.txt", encoding = "UTF-8")
speech1 <- as.data.frame(speech1, stringsAsFactors = FALSE)
names(speech1) <- "sentence"
speech1[1:5,]

# If sentences are not separated
speech2 <- scan("Inaugural Address All.txt", what = "character", sep = "\n", encoding = "UTF-8")
speech2 <- strsplit(speech2, ". ", fixed = TRUE)
speech2 <- as.data.frame(speech2, stringsAsFactors = FALSE)
names(speech2) <- "sentence"
speech2[1:5,]

# Read Data from a Text file (.pdf)
# Convert PDF to TXT
file <- "C:\\Temp\\FOMC_LongerRunGoals.pdf"
exe <- "C:\\xpdfbin-win-3.04\\bin64\\pdftotext.exe"
system(paste("\"", exe, "\" \"", file, "\"", sep = ""), wait = F)

fomc <- paste(readLines("C:\\Temp\\FOMC_LongerRunGoals.txt"), collapse=" ")
fomc

fomc <- strsplit(fomc, ". ", fixed = TRUE)
fomc <- as.data.frame(fomc, stringsAsFactors = FALSE)
names(fomc) <- "sentence"
fomc[1:5,]


# Case 1-2: Import data from a DB table -----------------------------------

# ODBC Examples
install.packages("RODBC", dependencies = TRUE)
library(RODBC)

# Load the MS Access file
JournalData = odbcConnect("TM_Journal")
JournalData

# See the table descriptions
sqlTables(JournalData)

# Convert a table into a data frame
RawData = sqlFetch(JournalData, "Journal_Data")
str(RawData)

# Transform a variabe type from factor to char
RawData$Journal <- as.character(RawData$Journal)
RawData$Title <- as.character(RawData$Title)
RawData$Abstract <- as.character(RawData$Abstract)
str(RawData)


# Case 2-1: Collect Texts using Twitter API ---------------------------------
install.packages("twitteR")
install.packages("ROAuth")
install.packages("RCurl")
install.packages("streamR")
install.packages("rjson")
install.packages("base64enc")
install.packages("httr")

library(twitteR)
library(ROAuth)
library(RCurl)
library(streamR)
library(rjson)
library(base64enc)
library(httr)

consumer_key= "Your consumer key"
consumer_secret= "Your consumer secret"
access_token = "Your access token"
access_secret = "Your access scret"

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)

# Retrieve the first 100 tweets (or all tweets if fewer than 100)
# from the user timeline of @IBMWatson
WatsonTweets <- userTimeline("IBMWatson", n=100)
WatsonTweets[1:10]

# search research for the hashtag #AlphaGo
AlphaGoTweets <- searchTwitter("#AlphaGo", n=100)
AlphaGoTweets[1:10]

# search research for the hashtag #AlphaGo with time constraints
AlphaGoTweets2 <- searchTwitter("#AlphaGo", n=1000, 
                           since = '2017-02-01', until = '2017-02-6')
AlphaGoTweets2[1:10]

# Twitter stream data collection
download.file(url="http://curl.haxx.se/ca/cacert.pem", destfile="cacert.pem")

reqURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
apiKey <- "Your api key"
apiSecret <- "Your api secret"

twitCred <- OAuthFactory$new(consumerKey=apiKey, consumerSecret=apiSecret, 
                             requestURL=reqURL, accessURL=accessURL, authURL=authURL)

twitCred$handshake(cainfo ="cacert.pem")

filterStream(file="AI.json", track="Artificial Intelligence", language = "en", timeout=30, oauth=twitCred)

readFile <- file("AI.json", "r")
streamTweets <- readLines(readFile, -1L)

dfMentions <- data.frame()

for (i in 1:length(streamTweets)){
  dfMentions <- rbind(dfMentions, as.data.frame(fromJSON(streamTweets[i])$text))
}

# Case 2-2: Collect Texts using Facebook API ---------------------------------
install.packages("Rfacebook")
library(Rfacebook)

# Authentication Setting
my_oauth <- fbOAuth(app_id = "Your app id", app_secret= "Your app secret")
save(my_oauth, file = "my_oauth")
load("my_oauth")

# Check the numerical ID of a page with the follwing website: http://findmyfbid.com/

# Get data from a page
PageData <- getPage(245661478875011, token = my_oauth, n = 100)

# Get data from a group
GroupData <- getGroup(255834461424286, token = my_oauth, n = 100)

# Case 3: Web Scraping ----------------------------------------------------
install.packages("dplyr")
install.packages("stringr")
install.packages("httr")
install.packages("rvest")

library(dplyr)
library(stringr)
library(httr)
library(rvest)

url <- 'https://arxiv.org/find/all/1/all:+EXACT+text_mining/0/1/0/all/0/1?skip=0'
start <- proc.time()
title <- NULL
author <- NULL
subject <- NULL
abstract <- NULL
meta <- NULL

for( i in c(0,25,50,75,100,125,150)){
  
  tmp_url <- modify_url(url, query = list(skip = i))
  tmp_list <- read_html(tmp_url) %>% html_nodes('div#dlpage') %>% html_nodes('a[title="Abstract"]') %>% html_attr('href')
  tmp_list <- paste0('https://arxiv.org',tmp_list)
  
  for(j in 1:length(tmp_list)){
    
    tmp_paragraph <- read_html(tmp_list[j])
    
    # title
    tmp_title <- gsub('Title:\n', '',tmp_paragraph %>% html_nodes('h1.title.mathjax') %>% html_text(T))
    title <- c(title, tmp_title)
    
    # author
    tmp_author <- tmp_paragraph %>% html_nodes('div.authors') %>% html_text
    tmp_author <- gsub('\\s+',' ',tmp_author)
    tmp_author <- gsub('Authors:','',tmp_author) %>% str_trim
    author <- c(author, tmp_author)  
    
    # subject
    tmp_subject <- tmp_paragraph %>% html_nodes('span.primary-subject') %>% html_text(T)
    subject <- c(subject, tmp_subject)
    
    # abstract
    tmp_abstract <- tmp_paragraph %>% html_nodes('blockquote.abstract.mathjax') %>% html_text(T)
    tmp_abstract <- sub('Abstract:','',tmp_abstract) %>% str_trim
    abstract <- c(abstract, tmp_abstract)
    
    # meta
    tmp_meta <- tmp_paragraph %>% html_nodes('div.submission-history') %>% html_text
    tmp_meta <- lapply(strsplit(gsub('\\s+', ' ',tmp_meta), '[v1]', fixed = T),'[',2) %>% unlist %>% str_trim
    meta <- c(meta, tmp_meta)
    cat(j, "paper\n")
    
  }
  cat((i/25) + 1,'/ 7 page\n')
}
final <- data.frame(title, author, subject, abstract, meta)
end <- proc.time()
end - start # Total Elapsed Time

# Export the result
write.csv(final, file = "Text Mining arxiv papers.csv")
