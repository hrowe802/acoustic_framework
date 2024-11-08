library("dplyr")
library("tidyr")
library("splitstackshape")
gapData <- read.csv("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Gap_Data_For_R.csv")

# create empty dataframe
rm(gapTable)
gapTable = setNames(data.frame(matrix(ncol = 3)), c("Participant", "Task", "Gap"))

# declare variables
currentPt = gapData$Participant[[1]];
previousPt = gapData$Participant[[1]];
currentTask = gapData$Task[[1]];
previousTask = gapData$Task[[1]];
tableRow = 1;
VOTstart = gapData$Time[[1]];
currentTime = 0;
previousTime = 0;

# calculate duration of last timepoints in puh123 -> first timepoints in t123VOT, last timepoints in tuh123 -> first timepoints in k123VOT, and last timepoints in k12VOT -> first timepoints in p12VOT
for(myrow in 1:(nrow(gapData))){
  currentPt = as.character(gapData$Participant[[myrow]]); # set participant
  currentTask = as.character(gapData$Task[[myrow]]); # set task
  currentTime = gapData[myrow, "Time"]; # set current time
  # if this is kuh3, gap is last timepoint in kuh3 plus average of kuh1gap and kuh2gap
  if(currentTask == "p2VOT" & previousTask == "kuh1"){
    kuh1gap = currentTime - previousTime;
  }
  if(currentTask == "p3VOT" & previousTask == "kuh2"){
    kuh2gap = currentTime - previousTime;
  }
  # if this is a new VOT task and not p1VOT, calculate gap
  if(currentTask != previousTask & (grepl("VOT", currentTask) & !grepl("p1VOT", currentTask))){
    Gap = currentTime - previousTime;
    # write out the results to the new table
    gapTable[tableRow, 1] = previousPt; # enter participant into table
    gapTable[tableRow, 2] = previousTask; # enter task into table
    gapTable[tableRow, 3] = Gap; # enter duration into table
    tableRow = tableRow + 1; # advance table row
    # reset variables for next iteration
    previousTime = currentTime;
  }
  # last row is a kuh, so use last timepoint in kuh3 plus average of kuh1gap and kuh2gap
  if((myrow == nrow(gapData)) | (currentTask == "p1VOT" & previousTask == "kuh3")){
    if(myrow == nrow(gapData)){
      lastKuh = gapData[myrow, "Time"];
      kuhPt = currentPt;
    } else {
      lastKuh = gapData[myrow-1, "Time"];
      kuhPt = gapData[myrow-1, "Participant"];
    }
    Gap = mean(c(kuh1gap, kuh2gap));
    # write out the results to the new table
    gapTable[tableRow, 1] = kuhPt; # enter participant into table
    gapTable[tableRow, 2] = "kuh3"; # enter task into table
    gapTable[tableRow, 3] = Gap; # enter duration into table
    tableRow = tableRow + 1; # advance table row
    # reset variables for next iteration
    previousTime = currentTime;
  }
  previousPt = currentPt;
  previousTask = currentTask;
  previousTime = currentTime;
}

# add column of rep
gapTable$Rep <- list()
for(i in 1:length(gapTable$Participant)){
  if(grepl("1", gapTable$Task[i])){
    gapTable$Rep[i] <- "1"
  } else if(grepl("2", gapTable$Task[i])){
    gapTable$Rep[i] <- "2"
  } else if(grepl("3", gapTable$Task[i])){
    gapTable$Rep[i] <- "3"
  }
  c(gapTable$Rep, gapTable$Rep[i])
}

# delete number from task
gapTable <- as.data.frame(gapTable)
gapTable$Task <- gsub('.{1}$', '', gapTable$Task)

# reorder task so it goes kuh puh tuh
for(i in 1:length(gapTable$Task)){
  if(i %% 3 == 0){
    puhRow = gapTable[i-2,]
    tuhRow = gapTable[i-1,]
    kuhRow = gapTable[i,]
    gapTable[i-2,] = kuhRow
    gapTable[i-1,] = puhRow
    gapTable[i,] = tuhRow
  }
}

# write to csv
gapTable <- write.csv(gapTable, file = "/Users/hannahrowe/Desktop/Gap_Data_For_Spreadsheets.csv")
