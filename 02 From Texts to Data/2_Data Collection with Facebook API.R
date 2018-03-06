# Case 1-2: Collect Texts using Facebook API ---------------------------------
install.packages("Rfacebook")
library(Rfacebook)

# Authentication Setting
my_oauth <- fbOAuth(app_id = "104477360011407", app_secret= "aeab0b819ddcbf4bca2decab0ba22878")
save(my_oauth, file = "my_oauth")
load("my_oauth")

# Check the numerical ID of a page with the follwing website: http://findmyfbid.com/

# Get data from a page (Bamboo Forest for KU Students)
# https://www.facebook.com/koreabamboo/?fref=ts

PageData <- getPage(206910909512230, token = my_oauth, n = 100)
write.csv(PageData, file = "Bamboo_KU.csv")

# Get data from a group (Deep learning group)
# https://www.facebook.com/groups/TensorFlowKR/

GroupData <- getGroup(255834461424286, token = my_oauth, n = 100)
write.csv(GroupData, file = "Tensorflow_KR.csv")
