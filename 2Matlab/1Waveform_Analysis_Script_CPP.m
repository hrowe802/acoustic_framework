% import audio
[data, fs] = audioread('/Users/hannahrowe/Google Drive/Research/Text Grids/Text Grids (Rate)/Fast_Hannah_DDK.wav');

% detrend signal to bring waveform down to zero line
dataDetrend = detrend(data, 'constant');

% timeline of data
time = (0:length(data)-1)/fs;

subplot(1,5,1)
     plot(time, dataDetrend)
     legend('Waveform (Detrended)')
     xlabel('Time')
     ylabel('Amplitude')

% rectify signal to make all negative values positive
dataRectify = abs(dataDetrend);

subplot(1,5,2)
     plot(time, dataRectify)
     legend('Waveform (Rectified)')
     xlabel('Time')
     ylabel('Amplitude')

% apply low pass filter to get rid of voice signal (source)
fc = 30;
fcc = fc/(fs/2);
[fb,fa] = butter(6,fcc);
dataLPF = filtfilt(fb,fa,dataRectify);

subplot(1,5,3)
     plot(time, dataLPF)
     legend('Waveform (Low Pass Filtered)')
     xlabel('Time')
     ylabel('Amplitude')

% apply fourier transform to data
dataFourierLPF = fft(dataLPF.*hamming(length(dataLPF)));
hz5000 = 5000 * length(dataFourierLPF) / fs;
frequency = (0:hz5000) * fs / length(dataFourierLPF);
[n10Hz  m10Hz] = find(frequency < 14);
maxFreqFFT = m10Hz(end);
dataFourierLPFlog = 20 * log10(abs(dataFourierLPF(1:length(frequency))) + eps);
dataTemp = dataFourierLPFlog(1:maxFreqFFT);

subplot(1,5,4)
    plot(frequency(1:maxFreqFFT), dataTemp)
    legend('Spectrum')
    xlabel('Frequency')
    ylabel('Magnitude')

% calculate cepstrum using spectrum
dataCepstrumLPF = fft(log(abs(dataFourierLPF) + eps));
maxFreqCep = fs/10; % maximum speech frequency at 10
minFreqCep = fs/.5; % minimum speech frequency at .5
quefrency = (maxFreqCep:minFreqCep)/fs;

subplot(1,5,5)
	plot(quefrency, abs(dataCepstrumLPF(maxFreqCep:minFreqCep)))
	legend('Cepstrum')
	xlabel('Quefrency')
	ylabel('Power')

% find max Y and corresponding X value
[peaksY peaksX] = findpeaks(abs(dataCepstrumLPF(maxFreqCep:minFreqCep)), quefrency);
[maxPeakY maxPeakI] = max(peaksY);
maxPeakX = peaksX(maxPeakI);

% fit linear regression line
linReg = fitlm(quefrency, abs(dataCepstrumLPF(maxFreqCep:minFreqCep)));

% get value of regression line at max Y
linRegValue = predict(linReg, maxPeakX);

% subtract regression value from max Y
CPP = maxPeakY - linRegValue
