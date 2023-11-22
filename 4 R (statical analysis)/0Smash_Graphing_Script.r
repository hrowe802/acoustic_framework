library("readxl")
library("ggplot2")
library("Hmisc")
library("dplyr")
library("ggsignif")
library("rlist")
library("effsize")
library("fmsb")
library("plotly")
library("ggpubr")
library("plotROC")
library("pROC")
library("readr")
library("psych")
library("mosaic")
library("car")
library("tidyr")
library("apaTables")
library("arsenal")
library("survival")
library("knitr")
library("ez")
library("nlme")
library("lme4")
library("lsmeans")
library("lmerTest")
library("tidyverse")
library("cluster")
library("factoextra")
library("Ckmeans.1d.dp")
library("lsr")
library("plyr")
library("afex")
library("emmeans")
dataSmash <- readxl::read_excel("/Users/hannahrowe/Google\ Drive/Research/Projects/RATE/RATE2_Data.xlsx")

# delete all rows with na
dataSmash <- na.omit(dataSmash)

# add column of group (only for rate data)
dataSmash$Group <- list()
for(i in 1:length(dataSmash$Filename)){
  if(grepl("Fast", dataSmash$Filename[i])){
    dataSmash$Group[i] <- "Fast"
  } else if(grepl("Normal", dataSmash$Filename[i])){
    dataSmash$Group[i] <- "Normal"
  } else if(grepl("Slow", dataSmash$Filename[i])){
    dataSmash$Group[i] <- "Slow"
  }
  c(dataSmash$Group, dataSmash$Group[i])
}

# add column of rep
dataSmash$Rep <- list()
for(i in 1:length(dataSmash$Filename)){
  if(grepl("R1.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "1"
  } else if(grepl("R2.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "2"
  } else if(grepl("R3.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "3"
  } else if(grepl("R4.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "4"
  } else if(grepl("R5.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "5"
  } else if(grepl("R6.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "6"
  } else if(grepl("R7.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "7"
  } else if(grepl("R8.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "8"
  } else if(grepl("R9.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "9"
  } else if(grepl("R10.txt", dataSmash$Filename[i])){
    dataSmash$Rep[i] <- "10"
  }
  c(dataSmash$Rep, dataSmash$Rep[i])
}

# add column of person (only for rate data)
dataSmash$Person <- list()
for(i in 1:length(dataSmash$Filename)){
  if(grepl("Brian", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Brian"
  } else if(grepl("Hannah", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Hannah"
  } else if(grepl("Hayden", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Hayden"
  } else if(grepl("Kaila", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Kaila"
  } else if(grepl("Sarah", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Sarah"
  } else if(grepl("Victoria", dataSmash$Filename[i])){
    dataSmash$Person[i] <- "Victoria"
  }
  c(dataSmash$Person, dataSmash$Person[i])
}

# add column of marker type (only for rate data)
dataSmash$Type <- list()
for(i in 1:length(dataSmash$Marker)){
  if(grepl("-", dataSmash$Marker[i])){
    dataSmash$Type[i] <- "Distance"
  } else(dataSmash$Type[i] <- "Individual")
    c(dataSmash$Type, dataSmash$Type[i])
}

# convert all character types to numeric
dataSmash$Rep <- as.numeric(as.character(dataSmash$Rep))

# make new dataset with means
smashMeans <- dataSmash %>%
  dplyr::group_by(Person, Group, Type, Marker, Signal_Tongue, Signal_Lips) %>%
  dplyr::summarise(TemplateSimilarity = mean(TemplateSimilarity), # STI
                   RepetitionSimilarity = mean(RepetitionSimilarity), # STI
                   STI = mean(STI), # STI
                   NJC = mean(NJC), # STI
                   LCoeff_Tongue = mean(LCoeff_Tongue), # STC
                   LCoeff_Lips = mean(LCoeff_Lips), # STC
                   Zcorr_Tongue = mean(Zcorr_Tongue), # STC
                   Zcorr_Lips = mean(Zcorr_Lips)) # STC

# subset by group (only means)
fast_means <- subset(smashMeans, Group == "Fast")
slow_means <- subset(smashMeans, Group == "Slow")
normal_means <- subset(smashMeans, Group == "Normal")

# subset by group (all data points)
fast <- subset(dataSmash, Group == "Fast")
slow <- subset(dataSmash, Group == "Slow")
normal <- subset(dataSmash, Group == "Normal")

# subset by type (only means)
individual_means <- subset(smashMeans, Type == "Individual")
distance_means <- subset(smashMeans, Type == "Distance")

# subset by type (all data points)
individual <- subset(dataSmash, Type == "Individual")
distance <- subset(dataSmash, Type == "Distance")

# statistics
res.aov <- aov(NJC ~ Group, data = subset(smashMeans, Marker == "UL_-LL_d"))
summary(res.aov)
TukeyHSD(res.aov)

# boxplot
ggplot(data = subset(smashMeans, Marker == "UL_-LL_d" | Marker == "UL_d" | Marker == "LL_d"),
       aes(x = Marker,
           y = RepetitionSimilarity,
           fill = Type)) +
  geom_boxplot() +
  geom_point() +
  facet_wrap(~Group) +
  labs(x = "Marker",
       y = "Similarity in Lip Distance Between Repetitions")

# scatterplot
ggplot(data = subset(smashMeans, Marker == "UL_-LL_da"),
       aes(x = TemplateSimilarity,
           y = RepetitionSimilarity,
           color = Group)) +
  geom_point(size = 4) +
  stat_cor(method = "pearson",
           hjust = -0.7,
           vjust = 1.5,
           family = "Times New Roman") + # C0RRELATION
  labs(x = "CONSISTENCY \n (Similarity in Lip Acceleration Compared to Template)",
       y = "COORDINATION \n (Similarity in Lip Acceleration Between Repetitions)")
