% loop through all audio files in folder
myDir = '/Users/hannahrowe/Google Drive/Research/Acoustic Files/Analyzed/Text Grids (TBI-HC) AMR/'; % get directory
myFiles = dir(fullfile(myDir,'*.wav')); % get all wav files in struct
SPPlist = [];
subject = [];
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  [data, fs] = audioread(fullFileName); % import audio
  
  dataDetrend = detrend(data, 'constant'); % detrend signal to bring waveform down to zero line
  time = (0:length(data)-1)/fs; % timeline of data
  subplot(1,4,1) % plot data (if needed)
     plot(time, dataDetrend)
     legend('Waveform (Detrended)')
     xlabel('Time')
     ylabel('Amplitude')
  
  dataRectify = abs(dataDetrend); % rectify signal to make all negative values positive
  subplot(1,4,2) % plot data (if needed)
     plot(time, dataRectify)
     legend('Waveform (Rectified)')
     xlabel('Time')
     ylabel('Amplitude')
  
  fc = 30; % apply low pass filter to get rid of voice signal (source)
  fcc = fc/(fs/2);
  [fb,fa] = butter(6,fcc);
  dataLPF = filtfilt(fb,fa,dataRectify);
  subplot(1,4,3) % plot data (if needed)
     plot(time, dataLPF)
     legend('Waveform (Low Pass Filtered)')
     xlabel('Time')
     ylabel('Amplitude')

  dataFourierLPF = fft(dataLPF.*hamming(length(dataLPF))); % apply fourier transform to data
  hz5000 = 5000 * length(dataFourierLPF) / fs;
  frequency = (0:hz5000) * fs / length(dataFourierLPF);
  [n10Hz  m10Hz] = find(frequency < 14);
  maxFreqFFT = m10Hz(end);
  dataFourierLPFlog = 20 * log10(abs(dataFourierLPF(1:length(frequency))) + eps);
  dataTemp = dataFourierLPFlog(1:maxFreqFFT);
  subplot(1,4,4) % plot data (if needed)
    plot(frequency(1:maxFreqFFT), dataTemp)
    legend('Spectrum')
    xlabel('Frequency')
    ylabel('Magnitude')

  [peaksY peaksX] = findpeaks(dataTemp, frequency(1:maxFreqFFT)); % find max Y and corresponding X value
  [maxPeakY maxPeakI] = max(peaksY);
  maxPeakX = peaksX(maxPeakI);
  linReg = fitlm(frequency(1:maxFreqFFT), dataTemp); % fit linear regression line
  linRegValue = predict(linReg, maxPeakX); % get value of regression line at max Y
  SPP = maxPeakY - linRegValue % subtract regression value from max Y
  SPPlist = [SPPlist, SPP];
  subject = [subject; baseFileName];
end

SPPlist = SPPlist'
