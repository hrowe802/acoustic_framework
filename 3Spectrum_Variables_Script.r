library("data.table")

spectrumData <- as.data.table(read.table("/Users/hannahrowe/Google\ Drive/My\ Drive/Research/Scripts/4\ R\ Scripts/Spectrum_Data_For_R.csv", header=TRUE, sep=","))

precision <- spectrumData[,
                         list(Task,
                              CentralGravity,
                              StandardDeviation,
                              Skewness,
                              Kurtosis,
                              PhonVar_CentralGravity = sd(CentralGravity),
                              PhonVar_StandardDeviation = sd(StandardDeviation),
                              PhonVar_Skewness = sd(Skewness),
                              PhonVar_Kurtosis = sd(Kurtosis)),
                         by = list(Participant, Rep)]

precision_consistency <- precision[,
                                   list(Rep,
                                        CentralGravity,
                                        StandardDeviation,
                                        Skewness,
                                        Kurtosis,
                                        PhonVar_CentralGravity,
                                        PhonVar_StandardDeviation,
                                        PhonVar_Skewness,
                                        PhonVar_Kurtosis,
                                        RepVar_CentralGravity = (sd(CentralGravity)/(mean(CentralGravity)))*100,
                                        RepVar_StandardDeviation = (sd(StandardDeviation)/(mean(StandardDeviation)))*100,
                                        RepVar_Skewness = (sd(Skewness)/(mean(Skewness)))*100,
                                        RepVar_Kurtosis = (sd(Kurtosis)/(mean(Kurtosis)))*100),
                                   by = list(Participant, Task)]

spectrumData <- precision_consistency[,
                                      list(Task,
                                           CentralGravity,
                                           StandardDeviation,
                                           Skewness,
                                           Kurtosis,
                                           PhonVar_CentralGravity,
                                           PhonVar_StandardDeviation,
                                           PhonVar_Skewness,
                                           PhonVar_Kurtosis,
                                           RepVar_CentralGravity,
                                           RepVar_StandardDeviation,
                                           RepVar_Skewness,
                                           RepVar_Kurtosis),
                             by = list(Participant, Rep)]

write.csv(spectrumData, file = "/Users/hannahrowe/Desktop/Spectrum_Data_For_Spreadsheets.csv")
