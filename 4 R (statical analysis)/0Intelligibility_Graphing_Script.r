library("readxl")
library("ggplot2")
library("psych")
library("dplyr")
library("ggpubr")
library("sp")
library("tidyr")
library("spatialEco")
dataIntell <- readxl::read_excel("/Users/hannahrowe/Google\ Drive/Research/Projects/PHON/PHON_Data.xlsx")

######################################################################################################################
#################################################### CONVERTING DATATYPES ############################################
######################################################################################################################
# CONVERT ALL CHARACTER VECTORS INTO NUMERIC
dataIntell$perc_intell_words <- as.numeric(as.character(dataIntell$perc_intell_words))
dataIntell$perc_phon_correct <- as.numeric(as.character(dataIntell$perc_phon_correct))
options(scipen = 999)

# ADD ID COLUMN TO DATASET
dataIntell$ID <- seq.int(nrow(dataIntell))

# CONVERT VARIABLES SO ALL GOING IN SAME DIRECTION
# dataIntell$prec_ratings_num <- 100-(dataIntell$prec_ratings_num)
# dataIntell$intell_ratings_num <- 100-(dataIntell$intell_ratings_num)
# dataIntell$effort_ratings <- 100-(dataIntell$effort_ratings)

# EXCLUDE ALL MISSING DATA
dataIntell = na.omit(dataIntell)

######################################################################################################################
#################################################### STRATIFYING #####################################################
######################################################################################################################
# ADD COLUMN FOR PROGRESSIVE VERSUS NONPROGRESSIVE ACCORDING TO INTELL AND PPC
dataIntell$Group <- list()
for(i in 1:length(dataIntell$Group)){
  if(dataIntell$Participant[i] == "DA003" |
     dataIntell$Participant[i] == "DA004" |
     dataIntell$Participant[i] == "DA007" |
     dataIntell$Participant[i] == "DA011" |
     dataIntell$Participant[i] == "DA014" |
     dataIntell$Participant[i] == "DA019" |
     dataIntell$Participant[i] == "DA020" |
     dataIntell$Participant[i] == "MA001" |
     dataIntell$Participant[i] == "MA005"){
    dataIntell$Group[i] <- "nonprogressive"
  } else if(dataIntell$Participant[i] == "0085" |
            dataIntell$Participant[i] == "0188" |
            dataIntell$Participant[i] == "0189" |
            dataIntell$Participant[i] == "0190" |
            dataIntell$Participant[i] == "191" |
            dataIntell$Participant[i] == "0196" |
            dataIntell$Participant[i] == "0197" |
            dataIntell$Participant[i] == "0200" |
            dataIntell$Participant[i] == "127" |
            dataIntell$Participant[i] == "144" |
            dataIntell$Participant[i] == "149" |
            dataIntell$Participant[i] == "DA005" |
            dataIntell$Participant[i] == "DA006"){
    dataIntell$Group[i] <- "progressive"
  }
  c(dataIntell$Group, dataIntell$Group[i])
}

######################################################################################################################
#################################################### DATASETTING #####################################################
######################################################################################################################
# MAKE NEW DATASET WITH ONLY NECESSARY VARIABLES TO CREATE 95% CI OF PPC-INTELL RELATIONSHIP
dataIntell_95CI <- dataIntell[-c(3, 5:12, 14:16)]s

# MAKE LONG DATASET FOR INDIVIDUAL PLOTS OF EACH PARTICIPANT
dataIntell_abbrev <- dataIntell[-c(2, 6, 8, 11:12, 14:15)]
dataIntell_long <- dataIntell_abbrev %>%
  tidyr::gather(key = "Variable", value = "Value", perc_intell_words:perc_phon_correct)

######################################################################################################################
################################################### SPAGHETTI PLOTS ##################################################
######################################################################################################################
# SPAGHETTI PLOT OF ALL PARTICIPANTS
ggplot(data = dataIntell,
       aes(x = days_since_first_session,
           y = perc_intell_words,
           color = Participant)) +
  geom_line() +
  labs(title = "Intelligibility (ORTHO) Over Time") +
  theme(plot.title = element_text(hjust = 0.5,
                                  face = "bold"))

# SPAGHETTI PLOT OF EACH PARTICIPANT
MA005 <- subset(dataIntell_long, Participant == "MA005")
ggplot(data = MA005,
       aes(x = days_since_first_session,
           y = Value,
           color = Variable)) +
  geom_line() +
  labs(title = "MA005 (Prog)") +
  theme(plot.title = element_text(size = 22,
                                  hjust = 0.5,
                                  face = "bold"),
        legend.title = element_text(size = 22),
        legend.text = element_text(size = 14),
        axis.title.x = element_text(size = 22),
        axis.text.x = element_text(size = 18),
        axis.title.y = element_text(size = 22),
        axis.text.y = element_text(size = 18))

######################################################################################################################
#################################################### SCATTERPLOTS ####################################################
######################################################################################################################
# SCATTERPLOT MATRIX OF ALL PARTICIPANTS
psych::pairs.panels(dataIntell[c(4,5,7)],
                    method = "pearson", # correlation method
                    density = TRUE,  # show density plots
                    ellipses = FALSE, # show correlation ellipses
                    lm = TRUE, # fit linear model curve to data
                    ci = TRUE) # add confidence interval

# SCATTERPLOT OF GROUP STRATIFICATIONS
ggplot(data = dataIntell,
       aes(x = perc_phon_correct,
           y = perc_intell_words)) +
  geom_point() +
  geom_smooth(method = "loess") +
  facet_wrap(~Group) +
  stat_cor(method = "pearson",
           label.y = 120) +
  labs(title = "Precision (PPC) by Precision (VAS) in Progressors versus Nonprogressors") +
  theme(plot.title = element_text(hjust = 0.5,
                                  face = "bold"))

# SCATTERPLOT OF NO STRATIFICATIONS
ggplot(data = dataIntell,
       aes(x = perc_phon_correct,
           y = perc_intell_words,
           color = Session)) +
  geom_point() +
  geom_smooth(method = "loess") +
  stat_cor(method = "pearson",
           label.y = 120) +
  labs(title = "Precision (PPC) by Precision (VAS)") +
  theme(plot.title = element_text(hjust = 0.5,
                                  face = "bold")) +
  geom_text(aes(label = Participant),
            hjust = 0.5,
            vjust = -0.8,
            size = 2)

######################################################################################################################
################################################### BUBBLE PLOTS #####################################################
######################################################################################################################
# BUBBLE PLOT OF GROUP STRATIFICATIONS
ggplot(data = dataIntell,
       aes(x = perc_phon_correct,
           y = perc_intell_words,
           size = words_per_min,
           fill = Session)) +
  facet_wrap(~Group) +
  geom_point(alpha = 0.5,
             shape = 21,
             color = "black") +
  scale_size(range = c(.5, 10)) +
  facet_wrap(~Group)

# BUBBLE PLOT OF NO STRATIFICATIONS
ggplot(data = dataIntell,
       aes(x = perc_phon_correct,
           y = perc_intell_words,
           size = words_per_min,
           fill = Session)) +
  geom_point(alpha = 0.5,
             shape = 21,
             color = "black") +
  scale_size(range = c(.5, 10)) +
  facet_wrap(~Group)

######################################################################################################################
################################################# POINTS IN POLYGON ##################################################
######################################################################################################################
# 0: point is strictly exterior to polygon
# 1: point is strictly interior to polygon
# 2: point lies on the relative interior of an edge of polygon

# IDENTIFY POINTS INSIDE AND OUTSIDE CI
ggplot(data = dataIntell_95CI,
       aes(x = perc_phon_correct,
           y = perc_intell_words)) +
  geom_point() +
  geom_smooth() -> gg

# DO CALCULATIONS
gb <- ggplot_build(gg)

# GET CI DATA
p <- gb$dataIntell[[2]]

# MAKE POLYGON OUT OF CI DATA
poly <- data.frame(x = c(p$x[1], p$x, p$x[length(p$x)], rev(p$x)),
                   y = c(p$ymax[1], p$ymin, p$ymax[length(p$x)], rev(p$ymax)))

# TEST FOR ORIGINAL VALUES IN POLYGON AND ADD TO ORIGINAL DATA
dataIntell_95CI$in_ci <- sp::point.in.polygon(dataIntell_95CI$perc_phon_correct,
                                          dataIntell_95CI$perc_intell_words,
                                          poly$x,
                                          poly$y)

# REDO PLOT WITH NEW DATA
ggplot(data = dataIntell_95CI,
       aes(x = perc_phon_correct,
           y = perc_intell_words)) +
  geom_point(aes(color = factor(in_ci))) +
  geom_smooth() +
  geom_text(aes(label = Participant),
            hjust = 0.5,
            vjust = -0.8,
            size = 2)

######################################################################################################################
##################################################### STATISTICAL MODELS #############################################
######################################################################################################################
# KEY CONSTRUCTS (SIMULTANEOUS REGRESSION MODELS)
# 1. What is the relationship between PPC and intelligibility?
model <- lm(data = dataIntell, perc_intell_words ~ perc_phon_correct)
summary(model)
# R2 = 47.90% (p < .001)

# 2. What is the relationship between PPC and speaking rate?
model <- lm(data = dataIntell, words_per_min ~ perc_phon_correct)
summary(model)
# R2 = 49.66% (p < .001)

# 3. What is the relationship between PPC and intelligible speaking rate?
model <- lm(data = dataIntell, intell_words_per_min ~ perc_phon_correct)
summary(model)
# R2 = 48.72% (p < .001)

# MULTIVARIATE CLUSTERS (K MEANS CLUSTERING)
# Cluster errors that might be driving intelligibility (mean value for each category for each cluster)

# KET FEATURES (SIMULTANEOUS REGRESSION MODELS)
# What phonemic errors are the most deleterious to intelligibility?
