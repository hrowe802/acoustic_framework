clear all
close all
clc

%% Load data (CSP)
a=load('HR_CSP_3.mat');
aEMG=a.V190430_rowe_CSP_second_wave_data.values;
b=load('HR_CSP_5.mat');
bEMG=b.V190430_rowe_CSP_second_wave_data.values;
c=load('HR_CSP_6.mat');
cEMG=c.V190430_rowe_CSP_second_wave_data.values;
d=load('HR_CSP_8.mat');
dEMG=d.V190430_rowe_CSP_second_wave_data.values;
e=load('HR_CSP_10.mat');
eEMG=e.V190430_rowe_CSP_second_wave_data.values;
f=load('HR_CSP_11.mat');
fEMG=f.V190430_rowe_CSP_second_wave_data.values;
g=load('HR_CSP_12.mat');
gEMG=g.V190430_rowe_CSP_second_wave_data.values;
h=load('HR_CSP_13.mat');
hEMG=h.V190430_rowe_CSP_second_wave_data.values;
i=load('HR_CSP_14.mat');
iEMG=i.V190430_rowe_CSP_second_wave_data.values;
j=load('HR_CSP_15.mat');
jEMG=j.V190430_rowe_CSP_second_wave_data.values;

%% Set filter parameters (CSP)
Fs = 6400;  % Sampling Frequency
N   = 2;    % Order
Fc1 = 10;   % First Cutoff Frequency
Fc2 = 2000;  % Second Cutoff Frequency
h  = fdesign.bandpass('N,F3dB1,F3dB2', N, Fc1, Fc2, Fs);
Hd = design(h, 'butter');

%% Filter loaded data (CSP)
aMEP1=filter(Hd,aEMG(:,1)); % MEP channel:EMG1
bMEP1=filter(Hd,bEMG(:,1));
cMEP1=filter(Hd,cEMG(:,1));
dMEP1=filter(Hd,dEMG(:,1));
eMEP1=filter(Hd,eEMG(:,1));
fMEP1=filter(Hd,fEMG(:,1));
gMEP1=filter(Hd,gEMG(:,1));
hMEP1=filter(Hd,hEMG(:,1));
iMEP1=filter(Hd,iEMG(:,1));
jMEP1=filter(Hd,jEMG(:,1));

%% Pull all trials into one matrix (CSP)
all_trials = [aMEP1 bMEP1 cMEP1 dMEP1 eMEP1 fMEP1 gMEP1 hMEP1 iMEP1 jMEP1]; % Ch1

%% Get MEP average (CSP)
summation=(sum(all_trials,2));
MEPaverage=summation./10; % change #

%% Process EMG signal by averaging and rectifying EMG trace (CSP)
matricRec = abs(all_trials);
summation = (sum(matricRec,2));
EMGaverage = summation./10; % change #
EMGaverage_invert = EMGaverage';
preemg = EMGaverage(1:640,1);

%% Waterfall plot (CSP)
channel_n = size(all_trials);
SampInterval=1/6400;
x_axis = (1:length(aMEP1))*SampInterval;

for i = 1:channel_n(2)
    figure(1)
    plot(x_axis, all_trials(:,i) + 0.05*(i-1))
    title('MEPs of All CSP Trials')
    hold on
end
hold off

%% Offset calculation (CSP)
WinWidth = 10*6.4; % 10 ms moving window
for j = 1:1:length(EMGaverage_invert)-WinWidth+1
    SD_tenms(j) = std(EMGaverage_invert(j:j+WinWidth-1));
end

SD_trace = SD_tenms' + EMGaverage(1:length(SD_tenms));
baselineSD = mean(SD_trace(1:640));

%% Plot processed EMG trace (CSP)
figure(2)
plot(x_axis, EMGaverage, 'linewidth', 1.5, 'color',[0,0,0]+0.5), axis auto
title('FDI CSP')

hold on
plot((1:length(SD_tenms))*SampInterval, SD_tenms' + EMGaverage(1:length(SD_tenms)), 'b', 'linewidth',1)

hold on
plot(x_axis, baselineSD*ones(length(x_axis), 1))

hold off

%% Load data (SICI-ICF)
k=load('HR_SICIICF_all.mat');
kEMG=k.V190430_rowe_SICIICF_wave_data.values;

%% Calculate EMG amplitudes (SICI-ICF)
SICIICF=peak2peak(kEMG);
