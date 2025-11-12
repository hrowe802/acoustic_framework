library("data.table")
library("splitstackshape")

cepstrumData <- as.data.table(read.table("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Cepstrum_Data_For_R.csv", header=TRUE,sep=","))

# make nine rows per observation
cepstrumData <- as.data.frame(cepstrumData)
cepstrumData <- splitstackshape::expandRows(cepstrumData, 3, count.is.col = FALSE)
cepstrumData <- apply(cepstrumData, 2, as.character)
cepstrumData <- write.csv(cepstrumData, file = "/Users/hannahrowe/Desktop/Cepstrum_Data_For_Spreadsheets.csv")
