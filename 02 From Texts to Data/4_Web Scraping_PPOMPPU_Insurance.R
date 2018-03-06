# Case 4: Web Scraping (arXiv Papers) -----------------------------------------
install.packages("dplyr")
install.packages("stringr")
install.packages("httr")
install.packages("rvest")

library(dplyr)
library(stringr)
library(httr)
library(rvest)

url <- 'http://www.ppomppu.co.kr/zboard/zboard.php?id=insurance&page='
start <- proc.time() # 
ppomppu_insurance <- data.frame()
Npost <- 1

# Extract the link of each post (for first 10 pages)
for( i in c(1:10)){
  
  tryCatch({
    
    tmp_url <- paste(url, i, '&divpage=10', sep="")
    tmp_list0 <- read_html(tmp_url) %>% html_nodes('tr.list0') %>% html_nodes('a') %>% html_attr('href')
    tmp_list1 <- read_html(tmp_url) %>% html_nodes('tr.list1') %>% html_nodes('a') %>% html_attr('href')
    tmp_list0 <- paste0('http://www.ppomppu.co.kr/zboard/',tmp_list0)
    tmp_list1 <- paste0('http://www.ppomppu.co.kr/zboard/',tmp_list1)
    tmp_list <- c(tmp_list0, tmp_list1)
    
    for(j in 1:length(tmp_list)){
      
      cat("Processing ", j, "-th Post of ", i, "-th page \n", sep="")
      
      tryCatch({
        tmp_paragraph <- read_html(tmp_list[j])
        
        # title
        tryCatch({
          tmp_title <- repair_encoding(tmp_paragraph %>% html_nodes('font.view_title2') %>% html_text(T))
        }, error = function(e){tmp_title <- NULL})
        
        # date
        tryCatch({
          tmp_date <- repair_encoding(tmp_paragraph %>% html_nodes('td.han') %>% html_text(T))[2]
          date_start_idx <- gregexpr(pattern ='등록일', tmp_date)[[1]][1]
          tmp_date <- substr(tmp_date, date_start_idx+5, date_start_idx+14)
        }, error = function(e){tmp_date <- NULL})
        
        # contents
        tryCatch({
          tmp_contents <- repair_encoding(tmp_paragraph %>% html_nodes('td.board-contents') %>% html_text(T))
          tmp_contents <- gsub("[[:punct:]]", " ", tmp_contents)
          tmp_contents <- gsub("[[:space:]]", " ", tmp_contents)
          tmp_contents <- gsub("\\s+", " ", tmp_contents)
          tmp_contents <- str_trim(tmp_contents, side = "both")
        }, error = function(e){tmp_contents <- NULL})
        
        ppomppu_insurance[Npost,1] <- tmp_title
        ppomppu_insurance[Npost,2] <- tmp_date
        ppomppu_insurance[Npost,3] <- tmp_contents
        Npost <- Npost + 1
        
      }, error = function(e){print("Invalid conversion, skip the post")})
    }
  }, error = function(e){print("Invalid conversion, skip the page")})
}

end <- proc.time()
end - start # Total Elapsed Time

# Export the result
write.csv(ppomppu_insurance, file = "ppomppu_insurance.csv")
