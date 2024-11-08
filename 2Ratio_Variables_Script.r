library("dplyr")
library("tidyr")
ratioData <- read.csv("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Ratio_Data_For_R.csv")

# create empty dataframe
ratioTable = setNames(data.frame(matrix(ncol = 3)), c("Participant", "Task", "BurstToBurst"))

# declare variables
currentPt = ratioData$Participant[[1]];
previousPt = ratioData$Participant[[1]];
currentTask = ratioData$Task[[1]];
previousTask = ratioData$Task[[1]];
tableRow = 1;
VOTstart = ratioData$Time[[1]];
currentTime = 0;
previousTime = 0;

# calculate duration of first timepoints in p123VOT -> t123VOT, t123VOT -> k123VOT, and k12VOT -> p12VOT
for(myrow in 1:(nrow(ratioData))){
  currentPt = as.character(ratioData$Participant[[myrow]]); # set participant
  currentTask = as.character(ratioData$Task[[myrow]]); # set task
  currentTime = ratioData[myrow, "Time"]; # set current time
  # if this is kuh2, save variables for when we calculate burst to burst for kuh3
  if(currentTask == "p2VOT" & previousTask == "kuh1"){
    kuh1gap = currentTime - previousTime;
  }
  if(currentTask == "p3VOT" & previousTask == "kuh2"){
    kuh2gap = currentTime - previousTime;
  }
  # if this is a new VOT task and not p1VOT, calculate burst to burst
  if(currentTask != previousTask & (grepl("VOT", currentTask) & !grepl("p1VOT", currentTask))){
    BurstToBurst = currentTime - VOTstart;
    # write out the results to the new table
    ratioTable[tableRow, 1] = previousPt; # enter participant into table
    ratioTable[tableRow, 2] = previousTask; # enter task into table
    ratioTable[tableRow, 3] = BurstToBurst; # enter duration into table
    tableRow = tableRow + 1; # advance table row
    # reset variables for next iteration
    VOTstart = currentTime;
  }
  # last row is a kuh, so use last timepoint in kuh3 minus first timepoints in k3VOT plus average of kuh1gap and kuh2gap
  if((myrow == nrow(ratioData)) | (currentTask == "p1VOT" & previousTask == "kuh3")){
    if(myrow == nrow(ratioData)){
      lastKuh = ratioData[myrow, "Time"];
      kuhPt = currentPt;
    } else {
      lastKuh = ratioData[myrow-1, "Time"];
      kuhPt = ratioData[myrow-1, "Participant"];
    }
    BurstToBurst = lastKuh - VOTstart;
    BurstToBurst = BurstToBurst + mean(c(kuh1gap, kuh2gap));
    # write out the results to the new table
    ratioTable[tableRow, 1] = kuhPt; # enter participant into table
    ratioTable[tableRow, 2] = "kuh3"; # enter task into table
    ratioTable[tableRow, 3] = BurstToBurst; # enter duration into table
    tableRow = tableRow + 1; # advance table row
    # reset variables for next iteration
    VOTstart = currentTime;
  }
  previousPt = currentPt;
  previousTask = currentTask;
  previousTime = currentTime;
}

# calculate ratio of puh->t / tuh->k
ratioTable$DurationRatio <- list()
for(i in 1:length(ratioTable$BurstToBurst)){
  if(i %% 3 == 0){
    # put same thing into all three rows so each task has one ratio
    ratioTable$DurationRatio[i-2] <- ratioTable[i-2,3] / ratioTable[i-1,3]
    ratioTable$DurationRatio[i-1] <- ratioTable[i-2,3] / ratioTable[i-1,3]
    ratioTable$DurationRatio[i] <- ratioTable[i-2,3] / ratioTable[i-1,3]
  }
c(ratioTable$DurationRatio, ratioTable$DurationRatio[i])
}

# rename task to puh, tuh, kuh
for(i in 1:length(ratioTable$Task)){
  if(ratioTable$Task[i] == "p1VOT"){
    ratioTable$Task[i] <- "puh1"
  } else if(ratioTable$Task[i] == "t1VOT"){
    ratioTable$Task[i] <- "tuh1"
  } else if(ratioTable$Task[i] == "k1VOT"){
    ratioTable$Task[i] <- "kuh1"
  } else if(ratioTable$Task[i] == "p2VOT"){
    ratioTable$Task[i] <- "puh2"
  } else if(ratioTable$Task[i] == "t2VOT"){
    ratioTable$Task[i] <- "tuh2"
  } else if(ratioTable$Task[i] == "k2VOT"){
    ratioTable$Task[i] <- "kuh2"
  } else if(ratioTable$Task[i] == "p3VOT"){
    ratioTable$Task[i] <- "puh3"
  } else if(ratioTable$Task[i] == "t3VOT"){
    ratioTable$Task[i] <- "tuh3"
  } else if(ratioTable$Task[i] == "k3VOT"){
    ratioTable$Task[i] <- "kuh3"
  }
}

# add column of rep
ratioTable$Rep <- list()
for(i in 1:length(ratioTable$Task)){
  if(grepl("1", ratioTable$Task[i])){
    ratioTable$Rep[i] <- "1"
  } else if(grepl("2", ratioTable$Task[i])){
    ratioTable$Rep[i] <- "2"
  } else if(grepl("3", ratioTable$Task[i])){
    ratioTable$Rep[i] <- "3"
  }
  c(ratioTable$Rep, ratioTable$Rep[i])
}

# delete number from task
ratioTable$Task <- gsub('.{1}$', '', ratioTable$Task)

# reorder task so it goes kuh puh tuh
for(i in 1:length(ratioTable$Task)){
  if(i %% 3 == 0){
    puhRow = ratioTable[i-2,]
    tuhRow = ratioTable[i-1,]
    kuhRow = ratioTable[i,]
    ratioTable[i-2,] = kuhRow
    ratioTable[i-1,] = puhRow
    ratioTable[i,] = tuhRow
  }
}

# add column of distance from 1
ratioTable$DistanceFrom1 <- list()
for(i in 1:length(ratioTable$DurationRatio)){
  ratioTable$DistanceFrom1[i] <- ratioTable$DurationRatio[i]-1
  c(ratioTable$DistanceFrom1, ratioTable$DistanceFrom1[i])
}

# move rep column over next to task
ratioTable <- ratioTable %>% relocate(Rep, .before = Task)

# write to csv
write.csv(ratioTable, file = "/Users/hannahrowe/Desktop/Ratio_Data_For_Spreadsheets.csv")
