library("data.table")
library("splitstackshape")

durationData <- as.data.table(read.table("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Duration_Data_For_R.csv", header=TRUE,sep=","))

# # TO FORMAT DATA FOR REP DURATION:
# # make three rows per observation
# durationData <- as.data.frame(durationData)
# durationData <- splitstackshape::expandRows(durationData, 3, count.is.col = FALSE)
# durationData <- apply(durationData, 2, as.character)
# durationData <- write.csv(durationData, file = "/Users/hannahrowe/Desktop/Duration_Data_For_Spreadsheets.csv")

# TO FORMAT DATA FOR FULL DURATION:
# make nine rows per observation
durationData <- as.data.frame(durationData)
durationData <- splitstackshape::expandRows(durationData, 9, count.is.col = FALSE)
durationData <- apply(durationData, 2, as.character)
durationData <- write.csv(durationData, file = "/Users/hannahrowe/Desktop/Duration_Data_For_Spreadsheets.csv")