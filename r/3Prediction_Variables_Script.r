library("data.table")
library("splitstackshape")

predictionData <- as.data.table(read.table("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Prediction_Data_For_R.csv", header=TRUE, sep=","))

# ADD NEW VARIABLES FOR BETWEEN-PHONEME DIFFERENCES
predictionData$ptk1_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$ptk1_sd[i] = sd(c(predictionData$p1[i], predictionData$t1[i], predictionData$k1[i]))}
c(predictionData$ptk1_sd, predictionData$ptk1_sd[i])

predictionData$ptk2_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$ptk2_sd[i] = sd(c(predictionData$p2[i], predictionData$t2[i], predictionData$k2[i]))}
c(predictionData$ptk2_sd, predictionData$ptk2_sd[i])

predictionData$ptk3_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$ptk3_sd[i] = sd(c(predictionData$p3[i], predictionData$t3[i], predictionData$k3[i]))}
c(predictionData$ptk3_sd, predictionData$ptk3_sd[i])

# ADD NEW VARIABLES FOR BETWEEN-REP DIFFERENCES
predictionData$p123_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$p123_sd[i] = sd(c(predictionData$p1[i], predictionData$p2[i], predictionData$p3[i]))}
c(predictionData$p123_sd, predictionData$p123_sd[i])

predictionData$t123_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$t123_sd[i] = sd(c(predictionData$t1[i], predictionData$t2[i], predictionData$t3[i]))}
c(predictionData$t123_sd, predictionData$t123_sd[i])

predictionData$k123_sd <- list()
for(i in 1:length(predictionData$Participant)){
  predictionData$k123_sd[i] = sd(c(predictionData$k1[i], predictionData$k2[i], predictionData$k3[i]))}
c(predictionData$k123_sd, predictionData$k123_sd[i])

# CONVERT NEW CHARACTER VECTORS INTO NUMERIC
predictionData$ptk1_sd <- as.numeric(as.character(predictionData$ptk1_sd))
predictionData$ptk2_sd <- as.numeric(as.character(predictionData$ptk2_sd))
predictionData$ptk3_sd <- as.numeric(as.character(predictionData$ptk3_sd))

predictionData$p123_sd <- as.numeric(as.character(predictionData$p123_sd))
predictionData$t123_sd <- as.numeric(as.character(predictionData$t123_sd))
predictionData$k123_sd <- as.numeric(as.character(predictionData$k123_sd))

# CREATE DATASET CONTAINING MEANS OF SPECTRAL SDS
sds_means <- predictionData[,
                        list(SDPrecisionSpecDiffMean = mean(c(ptk1_sd, ptk2_sd, ptk3_sd)),
                             SDConsistencySpecDiffMean = mean(c(p123_sd, t123_sd, k123_sd))),
                        by = list(Participant)]

# COMBINE DATASETS INTO ONE WITH NINE ROWS PER PARTICIPANT
spectralCorrelationIndex <- as.data.frame(sds_means)
spectralCorrelationIndex <- splitstackshape::expandRows(spectralCorrelationIndex, 9, count.is.col = FALSE)
spectralCorrelationIndex <- apply(spectralCorrelationIndex, 2, as.character)

write.csv(spectralCorrelationIndex, file = "/Users/hannahrowe/Desktop/Prediction_Data_For_Spreadsheets.csv")
