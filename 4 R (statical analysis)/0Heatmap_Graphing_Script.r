library("readr")
library("dplyr")
library("ggplot2")
library("psych")
library("mosaic")
library("car")
library("Hmisc")
library("tidyr")
library("MOTE")
library("ez")
library("lsr")
library("emmeans")
library("foreign")
library("apaTables")
library("plot3D")

heatmapDK_APB <- read.csv("/Users/hannahrowe/Desktop/DK_APB_Heatmap.csv", sep=",")
heatmapDK_FDI <- read.csv("/Users/hannahrowe/Desktop/DK_FDI_Heatmap.csv", sep=",")
heatmapKS_APB <- read.csv("/Users/hannahrowe/Desktop/KS_APB_Heatmap.csv", sep=",")
heatmapKS_FDI <- read.csv("/Users/hannahrowe/Desktop/KS_FDI_Heatmap.csv", sep=",")

DK_APB <- scatter3D(x = heatmapDK_APB$LocX,
                    y = heatmapDK_APB$LocY,
                    z = heatmapDK_APB$EMG,
                    pch = 16,
                    bty = "b2",
                    type = "h",
xlab = "
Posterior-Anterior Distance (cm)",
ylab = "
Medial-Lateral Distance (cm)",
zlab = "EMG Amplitude (μV)")

DK_FDI <- scatter3D(x = heatmapDK_FDI$LocX,
                    y = heatmapDK_FDI$LocY,
                    z = heatmapDK_FDI$EMG,
                    pch = 16,
                    bty = "b2",
                    type = "h",
xlab = "
Posterior-Anterior Distance (cm)",
ylab = "
Medial-Lateral Distance (cm)",
zlab = "EMG Amplitude (μV)")

KS_APB <- scatter3D(x = heatmapKS_APB$LocX,
                    y = heatmapKS_APB$LocY,
                    z = heatmapKS_APB$EMG,
                    pch = 16,
                    bty = "b2",
                    type = "h",
xlab = "
Posterior-Anterior Distance (cm)",
ylab = "
Medial-Lateral Distance (cm)",
zlab = "EMG Amplitude (μV)")

KS_FDI <- scatter3D(x = heatmapKS_FDI$LocX,
                    y = heatmapKS_FDI$LocY,
                    z = heatmapKS_FDI$EMG,
                    pch = 16,
                    bty = "b2",
                    type = "h",
xlab = "
Posterior-Anterior Distance (cm)",
ylab = "
Medial-Lateral Distance (cm)",
zlab = "EMG Amplitude (μV)")