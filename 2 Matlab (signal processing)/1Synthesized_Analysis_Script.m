% data_all =[];
%     fs = 44100; % sampling frequency 1 kHz
%     t = 0 : 1/fs : 20; % time scale
%     f = 200; % Hz, embedded dominant frequency
%
% for i = 1:1
%     data = cos(2*pi*i*t);
%     data_all = [data_all data];
% end

[data,fs] = audioread('/Users/hannahrowe/Google Drive/Research/Text Grids/Text Grids (Rate)/Slow_Hayden_DDK.wav');

% create synthesized waveform
%%%fs = 1000; % sampling frequency 1 kHz
%%%t = 0 : 1/fs : 0.296; % time scale
%%%f = 200; % Hz, embedded dominant frequency
%%%data = cos(2*pi*f*t);

% set parameters
x = data;
ms1 = fs/1000;                 % maximum speech Fx at 1000Hz
ms20 = fs/50;                  % minimum speech Fx at 50Hz

% plot waveform
t = (0:length(x) - 1) / fs;        % times of sampling instants
subplot(3,1,1);
plot(t, x);
legend('Waveform');
xlabel('Time (s)');
ylabel('Amplitude');

% do fourier transform of windowed signal
Y = fft(x.* hamming(length(x)));

% plot spectrum of bottom 5000Hz
hz5000 = 5000 * length(Y) / fs;
f = (0:hz5000) * fs / length(Y);
subplot(3,1,2);
plot(f, 20 * log10(abs(Y(1:length(f))) + eps));
legend('Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');

% cepstrum is DFT of log spectrum
C = fft(log(abs(Y) + eps));

% plot between 1ms (=1000Hz) and 20ms (=50Hz)
q = (ms1:ms20) / fs;
subplot(3,1,3);
plot(q, abs(C(ms1:ms20)));
legend('Cepstrum');
xlabel('Quefrency (s)');
ylabel('Amplitude');

% Fs = 1000; % sampling frequency 1 kHz
% t = 0 : 1/Fs : 0.296; % time scale
% f = 200; % Hz, embedded dominant frequency
% x = cos(2*pi*f*t) + randn(size(t)); % time series
% plot(t,x), axis('tight'), grid('on'), title('time series'), figure
% nfft = 512; % next larger power of 2
% y = fft(x,nfft); % Fast Fourier Transform
% y = abs(y.^2); % raw power spectrum density
% y = y(1:1+nfft/2); % half-spectrum
% [v,k] = max(y); % find maximum
% fScale = (0:nfft/2)* Fs/nfft; % frequency scale
% plot(fScale, y),axis('tight'),grid('on'),title('dominent frequency')
% fEst = fScale(k); % dominant frequency estimate
% fprintf('dominent frequency.: true %f Hz, estimated %f Hz\n', f, fEst)
% fprintf('frequency step (resolution) = %f Hz\n', fScale(2))
