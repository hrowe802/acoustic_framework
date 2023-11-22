library("readxl")
library("ggplot2")
library("Hmisc")
library("dplyr")
library("ggsignif")
library("rlist")
library("effsize")
library("fmsb")
library("plotly")
library("QuantPsyc")
library("ggpubr")
library("weights")
library("WRS2")
library("plotROC")
library("pROC")
library("ppcor")
library("readr")
library("psych")
library("mosaic")
library("car")
library("tidyr")
library("arsenal")
library("survival")
library("knitr")
library("lme4")
library("ez")
library("nlme")
library("lmerTest")
library("tidyverse")
library("egg")
library("cluster")
library("reshape")
library("factoextra")
library("Ckmeans.1d.dp")
library("lsr")
library("plyr")
library("afex")
library("devtools")
library("patchwork")
library("phaseR")
library("compute.es")
require("sos")
library("monitoR")
library("RColorBrewer")
library("extrafont")
library("wesanderson")
library("ggrepel")
library("rstatix")
library("gapminder")
library("pROC")
library("Epi")
library("ggbreak")

######################################################################################################################
########################################################### SETUP ####################################################
######################################################################################################################
# read in xlsx with all data

# CONVERT ALL CHARACTER VECTORS INTO NUMERIC
dataComp$DDKRate <- as.numeric(as.character(dataComp$DDKRate))
dataComp$PhonVar_CentralGravity <- as.numeric(as.character(dataComp$PhonVar_CentralGravity))
dataComp$PhonVar_StandardDeviation <- as.numeric(as.character(dataComp$PhonVar_StandardDeviation))
dataComp$PhonVar_Skewness <- as.numeric(as.character(dataComp$PhonVar_Skewness))
dataComp$PhonVar_Kurtosis <- as.numeric(as.character(dataComp$PhonVar_Kurtosis))
dataComp$RepVar_CentralGravity <- as.numeric(as.character(dataComp$RepVar_CentralGravity))
dataComp$RepVar_StandardDeviation <- as.numeric(as.character(dataComp$RepVar_StandardDeviation))
dataComp$RepVar_Skewness <- as.numeric(as.character(dataComp$RepVar_Skewness))
dataComp$RepVar_Kurtosis <- as.numeric(as.character(dataComp$RepVar_Kurtosis))

# ADD ID COLUMN TO DATASET
dataComp$ID <- seq.int(nrow(dataComp))
options(scipen = 999)

# CREATE KUH, ALLMEANS, KUHMEANS, PHONVAR, AND REPVAR DATASETS
puh <- subset(dataComp, Task == "puh")
tuh <- subset(dataComp, Task == "tuh")
kuh <- subset(dataComp, Task == "kuh")

# ENSURE THAT VARIABLES ARE GOING IN SAME DIRECTION
dataComp$RepVar_Syll <- -1*(dataComp$RepVar_Syll)
dataComp$RepVar_VOT <- -1*(dataComp$RepVar_VOT)
dataComp$F2Ratio <- -1*(dataComp$F2Ratio)
dataComp$GapDurationProp <- -1*(dataComp$GapDurationProp)
dataComp$VOT <- -1*(dataComp$VOT)
dataComp$DurationRatio <- -1*(dataComp$DurationRatio)
dataComp$DistanceFrom1 <- abs(dataComp$DistanceFrom1)
kuh$F2Slope <- abs(kuh$F2Slope)
kuh$CentralGravity <- -1*(kuh$CentralGravity)

# CREATE MEAN DATASETS
allMeans <- dataComp %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise_if(is.numeric, mean)

puhMeans <- puh %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise_if(is.numeric, mean)

tuhMeans <- tuh %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise_if(is.numeric, mean)

kuhMeans <- kuh %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise_if(is.numeric, mean)

phonVar <- dataComp %>%
  dplyr::group_by(Participant, Rep, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise(PhonVar_VOT = mean(PhonVar_VOT),
                   PhonVar_Vow = mean(PhonVar_Vow),
                   PhonVar_Syll = mean(PhonVar_Syll),
                   PhonVar_VOTVowProp = mean(PhonVar_VOTVowProp),
                   PhonVar_VOTSyllProp = mean(PhonVar_VOTSyllProp),
                   PhonVar_VowSyllProp = mean(PhonVar_VowSyllProp),
                   PhonVar_F1OnsetFreq = mean(PhonVar_F1OnsetFreq),
                   PhonVar_F2OnsetFreq = mean(PhonVar_F2OnsetFreq),
                   PhonVar_ConSpace = mean(PhonVar_ConSpace),
                   PhonVar_F1OffsetFreq = mean(PhonVar_F1OffsetFreq),
                   PhonVar_F2OffsetFreq = mean(PhonVar_F2OffsetFreq),
                   PhonVar_VowSpace = mean(PhonVar_VowSpace),
                   PhonVar_F1Range = mean(PhonVar_F1Range),
                   PhonVar_F2Range = mean(PhonVar_F2Range),
                   PhonVar_F1Slope = mean(PhonVar_F1Slope),
                   PhonVar_F2Slope = mean(PhonVar_F2Slope),
                   PhonVar_F1xF2Corr = mean(PhonVar_F1xF2Corr),
                   PhonVar_F1xF2Cov = mean(PhonVar_F1xF2Cov),
                   PhonVar_F1Ratio = mean(PhonVar_F1Ratio),
                   PhonVar_F2Ratio = mean(PhonVar_F2Ratio),
                   PhonVar_F1Vel = mean(PhonVar_F1Vel),
                   PhonVar_F1Accel = mean(PhonVar_F1Accel),
                   PhonVar_F1Jerk = mean(PhonVar_F1Jerk),
                   PhonVar_F2Vel = mean(PhonVar_F2Vel),
                   PhonVar_F2Accel = mean(PhonVar_F2Accel),
                   PhonVar_F2Jerk = mean(PhonVar_F2Jerk),
                   PhonVar_CentralGravity = mean(PhonVar_CentralGravity),
                   PhonVar_StandardDeviation = mean(PhonVar_StandardDeviation),
                   PhonVar_Skewness = mean(PhonVar_Skewness),
                   PhonVar_Kurtosis = mean(PhonVar_Kurtosis))

repVar <- dataComp %>%
  dplyr::group_by(Participant, Task, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise(RepVar_VOT = mean(RepVar_VOT),
                   RepVar_Vow = mean(RepVar_Vow),
                   RepVar_Syll = mean(RepVar_Syll),
                   RepVar_VOTVowProp = mean(RepVar_VOTVowProp),
                   RepVar_VOTSyllProp = mean(RepVar_VOTSyllProp),
                   RepVar_VowSyllProp = mean(RepVar_VowSyllProp),
                   RepVar_F1OnsetFreq = mean(RepVar_F1OnsetFreq),
                   RepVar_F2OnsetFreq = mean(RepVar_F2OnsetFreq),
                   RepVar_ConSpace = mean(RepVar_ConSpace),
                   RepVar_F1OffsetFreq = mean(RepVar_F1OffsetFreq),
                   RepVar_F2OffsetFreq = mean(RepVar_F2OffsetFreq),
                   RepVar_VowSpace = mean(RepVar_VowSpace),
                   RepVar_F1Range = mean(RepVar_F1Range),
                   RepVar_F2Range = mean(RepVar_F2Range),
                   RepVar_F1Slope = mean(RepVar_F1Slope),
                   RepVar_F2Slope = mean(RepVar_F2Slope),
                   RepVar_F1xF2Xcorr = mean(RepVar_F1xF2Xcorr),
                   RepVar_F1xF2Corr = mean(RepVar_F1xF2Corr),
                   RepVar_F1xF2Cov = mean(RepVar_F1xF2Cov),
                   RepVar_F1Ratio = mean(RepVar_F1Ratio),
                   RepVar_F2Ratio = mean(RepVar_F2Ratio),
                   RepVar_F1Vel = mean(RepVar_F1Vel),
                   RepVar_F1Accel = mean(RepVar_F1Accel),
                   RepVar_F1Jerk = mean(RepVar_F1Jerk),
                   RepVar_F2Vel = mean(RepVar_F2Vel),
                   RepVar_F2Accel = mean(RepVar_F2Accel),
                   RepVar_F2Jerk = mean(RepVar_F2Jerk),
                   RepVar_CentralGravity = mean(RepVar_CentralGravity),
                   RepVar_StandardDeviation = mean(RepVar_StandardDeviation),
                   RepVar_Skewness = mean(RepVar_Skewness),
                   RepVar_Kurtosis = mean(RepVar_Kurtosis))

absval <- dataComp %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise(F2Slope = abs(F2Slope))

absvalMeans <- absval %>%
  dplyr::group_by(Participant, Age, Gender, Group, Dataset, Diagnosis, Description, OverallSeverity, OverallSeverityCat, ArticSeverity, ArticSeverityCat) %>%
  dplyr::summarise(F2Slope = mean(F2Slope))

# RECODE TASKS TO BE NUMERIC
for(i in 1:length(repVar$Task)){
  if(repVar$Task[i] == "puh"){
    repVar$Task[i] <- 1
  } else if(repVar$Task[i] == "tuh"){
    repVar$Task[i] <- 2
  } else if(repVar$Task[i] == "kuh"){
    repVar$Task[i] <- 3
  } 
}
repVar$Task <- as.numeric(as.character(repVar$Task))

for(i in 1:length(dataComp$Task)){
  if(dataComp$Task[i] == "puh"){
    dataComp$Task[i] <- 1
  } else if(dataComp$Task[i] == "tuh"){
    dataComp$Task[i] <- 2
  } else if(dataComp$Task[i] == "kuh"){
    dataComp$Task[i] <- 3
  } 
}
dataComp$Task <- as.numeric(as.character(dataComp$Task))

# CREATE DATASETS CONSISTING OF EACH DIAGNOSIS
control <- subset(allMeans, Diagnosis == "Control")
control_kuh <- subset(kuhMeans, Diagnosis == "Control")
als <- subset(allMeans, Diagnosis == "ALS")
als_kuh <- subset(kuhMeans, Diagnosis == "ALS")
pa <- subset(allMeans, Diagnosis == "PA")
pa_kuh <- subset(kuhMeans, Diagnosis == "PA")
pd <- subset(allMeans, Diagnosis == "PD")
pd_kuh <- subset(kuhMeans, Diagnosis == "PD")
ppa <- subset(allMeans, Diagnosis == "PPA")
ppa_kuh <- subset(kuhMeans, Diagnosis == "PPA")

# CREATE DATASETS WITH ONLY VARIABLES OF INTEREST
dataQuant <- allMeans[,c("Participant", "Group", "Diagnosis", "Description", "RepVar_VOT", "GapDurationProp", "PhonVar_F2Slope", "DDKRate")]
speed <- kuhMeans$F2Slope
dataQuant$F2Slope <- speed

# CREATE INDIVIDUAL Z SCORES FOR DIAGNOSIS GROUPS (RELATIVE TO CONTROLS)
smd_dataQuant <- subset(dataQuant, Group == "SMD")
smd_dataQuant <- as.data.frame(smd_dataQuant)

smd_dataQuant$zRepVar_VOT <- list()
for(i in 1:length(smd_dataQuant$Participant)){
  smd_dataQuant$zRepVar_VOT[i] <- (smd_dataQuant$RepVar_VOT[i] - (mean(control$RepVar_VOT))) / sd(control$RepVar_VOT)}
c(smd_dataQuant$zRepVar_VOT, smd_dataQuant$RepVar_VOT[i])

smd_dataQuant$zGapDurationProp <- list()
for(i in 1:length(smd_dataQuant$Participant)){
  smd_dataQuant$zGapDurationProp[i] <- (smd_dataQuant$GapDurationProp[i] - (mean(control$GapDurationProp))) / sd(control$GapDurationProp)}
c(smd_dataQuant$zGapDurationProp, smd_dataQuant$zGapDurationProp[i])

smd_dataQuant$zDDKRate <- list()
for(i in 1:length(smd_dataQuant$Participant)){
  smd_dataQuant$zDDKRate[i] <- (smd_dataQuant$DDKRate[i] - (mean(control$DDKRate))) / sd(control$DDKRate)}
c(smd_dataQuant$zDDKRate, smd_dataQuant$zDDKRate[i])

smd_dataQuant$zF2Slope <- list()
for(i in 1:length(smd_dataQuant$Participant)){
  smd_dataQuant$zF2Slope[i] <- (smd_dataQuant$F2Slope[i] - (mean(control_kuh$F2Slope))) / sd(control_kuh$F2Slope)}
c(smd_dataQuant$zF2Slope, smd_dataQuant$zF2Slope[i])

smd_dataQuant$zPhonVar_F2Slope <- list()
for(i in 1:length(smd_dataQuant$Participant)){
  smd_dataQuant$zPhonVar_F2Slope[i] <- (smd_dataQuant$PhonVar_F2Slope[i] - (mean(control$PhonVar_F2Slope))) / sd(control$PhonVar_F2Slope)}
c(smd_dataQuant$zPhonVar_F2Slope, smd_dataQuant$zPhonVar_F2Slope[i])

# CREATE LONG DATASETS FOR PLOTS AND WIDE DATASETS FOR EFFECT SIZE CALCULATIONS
wide <- smd_dataQuant[,c(1:4,10:14)]
wide <- as.data.frame(wide)
long <- reshape::melt(wide,
                      id.vars = c("Participant", "Diagnosis", "Description"),
                      measure.vars = c("zGapDurationProp", "zRepVar_VOT", "zF2Slope", "zPhonVar_F2Slope", "zDDKRate"))
long <- rename(long, c("variable" = "Feature", "value" = "Value"))

zMeans_long <- long %>%
  dplyr::group_by(Diagnosis, Feature) %>%
  dplyr::summarise_if(is.numeric, mean)

zMeans_wide <- wide %>%
  dplyr::group_by(Diagnosis) %>%
  dplyr::summarise_if(is.numeric, mean)

# CLEANUP CODE
cleanup_for_combined_plots <- theme(plot.title = element_text(size = 24, hjust = 0.5, face = "bold", family = "Times New Roman"),
                                    panel.background = element_rect(fill = "white", color = "white", size = 0.5, linetype = "dotted"),
                                    panel.grid.major = element_line(size = 0.5, linetype = "dotted", colour = "gray"),
                                    panel.grid.minor = element_line(size = 0.25, linetype = "dotted", colour = "gray"),
                                    axis.title.x = element_text(size = 20, color = "black", face = "bold", family = "Times New Roman"),
                                    axis.title.y = element_text(size = 20, color = "black", face = "bold", family = "Times New Roman"),
                                    axis.text.x = element_text(size = 18, color = "black", family = "Times New Roman"),
                                    axis.text.y = element_text(size = 18, color = "black", family = "Times New Roman"),
                                    legend.title = element_blank(),
                                    legend.position = c(.2, .3),
                                    legend.text = element_text(size = 20, color = "black", family = "Times New Roman"))
cleanup_for_individual_plots <- theme(plot.title = element_text(size = 34, hjust = 0.5, face = "bold", family = "Times New Roman"),
                                      panel.background = element_rect(fill = "white", color = "white", size = 0.5, linetype = "dotted"),
                                      panel.grid.major = element_line(size = 0.5, linetype = "dotted", colour = "gray"),
                                      panel.grid.minor = element_line(size = 0.25, linetype = "dotted", colour = "gray"),
                                      axis.title.x = element_text(size = 26, color = "black", face = "bold", family = "Times New Roman"),
                                      axis.title.y = element_text(size = 26, color = "black", face = "bold", family = "Times New Roman"),
                                      axis.text.x = element_text(size = 24, color = "black", family = "Times New Roman"),
                                      axis.text.y = element_text(size = 24, color = "black", family = "Times New Roman"),
                                      legend.title = element_blank(),
                                      legend.position = c(.2, .3),
                                      legend.text = element_text(size = 24, color = "black", family = "Times New Roman"))

######################################################################################################################
##################################################### STATISTICAL MODELS #############################################
######################################################################################################################
# ANOVA MODELS (to determine differences in continuous outcome depending on categorical predictor with UNRELATED levels)
res.aov <- aov(DurationRatio ~ Diagnosis, data = allMeans)
summary(res.aov)
TukeyHSD(res.aov)
allMeans <- within(subset(allMeans, Description != "severe" & Description != "profound"), Diagnosis <- relevel(as.factor(Diagnosis), ref = "PD"))
res.lme <- lm(DDKRate ~ Diagnosis + (1 + Rep | as.numeric(Participant)), data = allMeans)
cohensD(DurationRatio ~ Diagnosis, data = allMeans)

# REPEATED MEASURES ANOVA (to determine differences in continuous outcome depending on categorical predictor with RELATED levels)
model <- ezANOVA(data = long, dv = .(Value), wid = .(Participant), within = .(Feature), type = 3, detailed = TRUE)
print(model)

# PAIRWISE or INDEPENDENT T TESTS (to determine where differences lie after running ANOVA)
pairwise <- pairwise.t.test(long$Value, long$Feature, paired = TRUE, p.adjust.method = "bonferroni")
pairwise

# EFFECT SIZES (to determine how big differences are after running PAIRWISE or INDEPENDENT T TESTS)
long %>%
  dplyr::group_by(Feature) %>%
  dplyr::summarise(mean = mean(Value), sd = sd(Value), n = n())
MOTE::d.dep.t.avg(-1.80, .03, 1.86, 1.36, 22, a = .05)

# MULTIPLE REGRESSION MODELS (to determine differences in CONTINUOUS outcome depending on continous predictors)
modelQuant_withOutcomes <- lm(SPEECH_Intell ~ RepVar_Syll + DDKRate + F2Slope + DurationRatio + PhonVar_F2Slope, data = dataQuant_withOutcomes)
summary(modelQuant_withOutcomes)

# LOGISTIC REGRESSION MODELS (to determine differences in CATEGORICAL outcome depending on continuous predictors)
allMeans$Group <- as.factor(allMeans$Group)
model <- glm(data = allMeans, Group ~ VOTSyllProp + DDKRate + PhonVar_F2Slope + RepVar_VOT, family = "binomial"(link = logit), na.action = na.exclude)
S(model)
# use model to predict each participant’s group membership
model.probs <- predict(model, type = "response")
head(model.probs)
# classify each participant as being in group or not
model.pred <- ifelse(model.probs > .5, "SMD", "Control")
model.pred <- factor(model.pred, levels = c("SMD", "Control"), labels = c("SMD", "Control"))
# confusion matrix (cross-tabs of predicted outcomes versus actual outcomes)
caret::confusionMatrix(data = model.pred, reference = allMeans$Group, positive = "SMD")

# LINEAR MIXED EFFECTS MODELS (to determine differences in continuous outcome depending on categorical predictor and controlling for repeated measures)
phonVar <- within(phonVar, Diagnosis <- relevel(as.factor(Diagnosis), ref = "Control"))
res.lme <- lm(RepVar_VOT ~ Group + (1 + Task | as.numeric(Participant)), data = repVar)
summary(res.lme)
effsize::cohen.d(ppa$DurationRatio, control$DurationRatio, conf.level = .90)
t.test(als$DurationRatio, control$DurationRatio)

# LINEAR MIXED EFFECTS MODELS WITH INTERACTION (to determine differences in continuous outcome depending on continous predictor at different categorical levels and controlling for repeated measures)
# run model to examine overall interaction between severity and component
model <- lme4::lmer(Value ~ Severity*Feature + (1 + Feature | Participant), data = long, REML = TRUE)
anova(model)
summary(model)
S(model) # interaction = effect of variable one on outcome depending on variable two
# run model to examine slopes
long$Feature <- as.factor(long$Feature)
model <- lme4::lmer(Value ~ Severity*Feature + (1 + Feature | Participant), data = long, REML = TRUE)
anova(model)
summary(model)
S(model)
# obtain slopes
modelSlopes <- lstrends(model, "Component", var = "Severity")
print(modelSlopes)
# compare slopes
modelTrends <- pairs(modelSlopes)
modelTrends

# CORRELATION MATRICES (to examine data before running statistical analyses)
which(colnames(allMeans_test)=="DurationRatio")
coordMatrix <- cbind(allMeans_test[124], allMeans_test[95], coordinationMeans_Ratio[2])
conMatrix <- cbind(allMeans_test[117], allMeans_test[61], consistencyMeans[2])
speedMatrix <- cbind(allMeans_test[123], kuhMeans_test[19], speedMeans[2])
precMatrix <- cbind(allMeans_test[121], allMeans_test[46], precisionMeans[2])
rateMatrix <- cbind(allMeans_test[115], allMeans_test[90], rateMeans[3])
psych::pairs.panels(coordMatrix, method = "pearson", lm = TRUE, hist.col = "turquoise", ellipses = FALSE, stars = TRUE, size = 6)

######################################################################################################################
################################################## SPAGHETTI PLOT ####################################################
######################################################################################################################
ggplot(data = zMeans_long_byDiagnosis, aes(x = Feature, y = Value, group = Diagnosis, color = Diagnosis)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  #geom_text(aes(label = Subject)) +
  cleanup +
  #facet_wrap(~Description) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = c(.8, .8))

######################################################################################################################
################################################## COLOR PALETTES ####################################################
######################################################################################################################
coolBluePalette <- c("darkslategray", "darkseagreen3", "cyan3", "cornflower blue", "deepskyblue4")

######################################################################################################################
#################################################### SCATTERPLOT #####################################################
######################################################################################################################
ggplot(data = bmiData, aes(x = eat10, y = bmi)) +
  geom_point(aes(shape = timepoint, color = bmilow), size = 4) +
  geom_smooth(method = "lm", size = 2, color = "cadetblue4", fill = "cadetblue3") +
  geom_line(aes(group = id, linetype = as.factor(bmichange_cat)), position = position_dodge(0.2)) +
  stat_ellipse(geom = "polygon", aes(fill = bmilow),
               alpha = 0.2,
               show.legend = FALSE, 
               level = 0.95,
               size = 2) +
  labs(title = "BMI as a Function of Baseline EAT-10 Scores",
       x = "Eat-10 Scores at Baseline",
       y = "BMI") +
  cleanup

######################################################################################################################
###################################################### BOXPLOT #######################################################
######################################################################################################################
ggplot(data = subset(allMeans, Description != "severe" & Description != "profound"), aes(x = Diagnosis, y = DDKRate, fill = Diagnosis)) +
  geom_boxplot() +
  geom_point(shape = 16, size = 3) +
  stat_summary(fun = mean, geom = "errorbar", aes(ymin = ..y.., ymax = ..y..), width = 0.75, linetype = 2) +
  #geom_text(aes(label = Participant), size = 2) +
  cleanup +
  theme(legend.position = "none")

######################################################################################################################
###################################################### BARPLOT #######################################################
######################################################################################################################
ggplot(data = subset(zscores_data, Group == "ALS" | Group == "PD"), aes(x = Feature, y = ZScore, fill = Group)) +
  geom_hline(yintercept = 0) +
  geom_bar(colour = "black", size = 1, stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("gray80", "gray50")) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, position = position_dodge(width = 0.9)) +
  scale_y_continuous(breaks = seq(-3.5, .7, .5), limits = c(-3.5, .7)) +
  labs(title = "Z Scores for ALS Versus PD", x = NULL, y = "Z Score") +
  cleanup +
  theme(legend.position = c(.9, .15), axis.text.x = element_text(angle = 10, size = 21))

######################################################################################################################
###################################################### ROC CURVE #####################################################
######################################################################################################################

# CREATE NEW DATASET WITH ONLY RELEVANT VARIABLES
ROCdata <- dataQuant %>%
  ungroup() %>%
  dplyr::select(Diagnosis, GapDurationProp, RepVar_VOT, F2Slope, CentralGravity, DDKRate)

# CONVERT DATA TO LONG FORMAT IN ORDER TO PUT MULTIPLE ROC CURVES ON ONE GRAPH
ROCdata <- melt_roc(ROCdata, d = "Diagnosis", m = c("GapDurationProp", "RepVar_VOT", "F2Slope", "CentralGravity", "DDKRate"))

# CREATE SUBSETS OF GROUPS TO COMPARE DIAGNOSIS GROUPS
als.pd <- subset(ROCdata, D == "ALS" | D == "PD")

# PLOT ROC CURVE
ROCcurve <- ggplot(als.pd, aes(d = D, m = M, color = name, linetype = name)) +
  geom_roc(n.cuts = 0, labels = FALSE) +
  style_roc(theme = theme_gray, xlab = "1 - Specificity", ylab = "Sensitivity") +
  ggtitle("ROC for ALS Versus PD") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = c(.8, .2)) +
  scale_color_manual(values = c("black", "slategray", "slategray3", "gray30", "gray60")) +
  scale_linetype_manual(values = c("dashed", "dotted", "dotdash", "longdash", "twodash")) +
  cleanup
ROCcurve

# CALCULATE AREA UNDER CURVE
calc_auc(ROCcurve)

# SENSITIVITY AND SPECIFICITY CUTPOINTS
# ALS versus Control
als.con_rate <- subset(als.pd, name == "Rate")
als.con_rate <- roc(als.pd_rate$D, als.con_rate$M)
coords(als.pd_rate, "best", ret = c("threshold", "specificity", "sensitivity", "accuracy"))

######################################################################################################################
######################################################## RADAR PLOT ##################################################
######################################################################################################################
zMeans_wide.allGroups <- zMeans_wide[c(1,8:12)]
zDurationRatio.Group1 <- zMeans_wide.allGroups[1,2]
zDurationRatio.Group2 <- zMeans_wide.allGroups[2,2]
zDurationRatio.Group3 <- zMeans_wide.allGroups[3,2]
zRepVar_Syll.Group1 <- zMeans_wide.allGroups[1,3]
zRepVar_Syll.Group2 <- zMeans_wide.allGroups[2,3]
zRepVar_Syll.Group3 <- zMeans_wide.allGroups[3,3]
zF2Slope.Group1 <- zMeans_wide.allGroups[1,4]
zF2Slope.Group2 <- zMeans_wide.allGroups[2,4]
zF2Slope.Group3 <- zMeans_wide.allGroups[3,4]
zPhonVar_F2Slope.Group1 <- zMeans_wide.allGroups[1,5]
zPhonVar_F2Slope.Group2 <- zMeans_wide.allGroups[2,5]
zPhonVar_F2Slope.Group3 <- zMeans_wide.allGroups[3,5]
zDDKRate.Group1 <- zMeans_wide.allGroups[1,6]
zDDKRate.Group2 <- zMeans_wide.allGroups[2,6]
zDDKRate.Group3 <- zMeans_wide.allGroups[3,6]

radarDiag = as.data.frame(matrix(c(zDurationRatio.Group1,
                                   zRepVar_Syll.Group1,
                                   zF2Slope.Group1,
                                   zPhonVar_F2Slope.Group1,
                                   zDDKRate.Group1),
  byrow = F, ncol = 5))
colnames(radarDiag) = c("Coordination", "Consistency            ", "Speed", "       Precision", "Rate")
rownames(radarDiag) = radarDiag$Severity
radarDiag <- rbind(rep(0.75, 5), rep(0.05, 5), radarDiag)
par(family = "Times New Roman", font = 2)

######################################################################################################################
#################################################### FOREST PLOT #####################################################
######################################################################################################################
# read in xlxs with list of effect sizes

forestplot_diagnosis <- forestplot_diagnosis %>%
  mutate(Component = factor(Component, levels = c("F1xF2Xcorr", "RepVar_SyllMean", "F2Slope", "PhonVar_F2Slope", "DDKRate"), labels = c("Coordination", "Consistency", "Speed", "Precision", "Rate")))
#levels(forestplot_diagnosis$Component) <- gsub("  ", "\n", levels(forestplot_diagnosis$Component))
ggplot(data = forestplot_diagnosis, aes(x = Component, y = D, ymin = Cilower, ymax = Ciupper)) +
  geom_pointrange(aes(col = Component)) +
  geom_hline(aes(fill = Component), yintercept = 0, linetype = 2) +
  geom_errorbar(aes(ymin = Cilower, ymax = Ciupper, col = Component), width = 0.5, cex = 1) +
  facet_wrap(~Group, strip.position = "left", nrow = 9) +
  theme(legend.position = c(0.85, 0.40),
        legend.background = element_rect(fill = "gray94"),
        legend.title = element_text(size = 20, face = "bold", family = "Times New Roman", hjust = .5),
        legend.text = element_text(size = 18, family = "Times New Roman"),
        panel.background = element_rect(fill = "white", color = "white", size = 0.5, linetype = "dotted"),
        panel.grid.major = element_line(size = 0.5, linetype = "dotted", color = "gray"), 
        panel.grid.minor = element_line(size = 0.25, linetype = "dotted", color = "gray"),
        plot.title = element_text(size = 26, face = "bold", family = "Times New Roman", hjust = .5),
        axis.title.x = element_text(size = 24, face = "bold", family = "Times New Roman", hjust = .5, vjust = .01),
        axis.title.y = element_blank(),
        axis.text.x = element_text(size = 20, color = "black", face = "bold", family = "Times New Roman"),
        axis.text.y = element_blank(),
        strip.text.x = element_text(hjust = 0, vjust = 10, angle = 180, size = 22, face = "bold", family = "Times New Roman"),
        strip.text.y = element_blank()) +
  labs(title = "ALS (Late Stage) versus PD (Late Stage)", x = "", y = "Cohen's d (95% CI)") +
  guides(color = guide_legend(reverse=T)) +
  scale_y_continuous(limits = c(-.2, 3)) +
  scale_color_manual(values = c("cornflowerblue", "tan2", "mediumpurple1", "darkslategray3", "palevioletred")) +
  #scale_color_manual(values = c("gray0", "gray75", "gray60", "gray40", "gray15")) +
  coord_flip()
