#!/usr/bin/env python3
"""
AMR/SMR Segmentation with Robust Noise Detection and Removal
Handles speech interference, background noise, and other artifacts
Supports both AMR (papapa, tatata, kakaka) and SMR (pataka) patterns
Now with batch processing and flexible file locations
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample, find_peaks, spectrogram, butter, filtfilt
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import tgt
from pathlib import Path
import argparse
import glob


class NoiseDetector:
    """Detects and removes various types of noise from DDK recordings"""

    def __init__(self, fs):
        self.fs = fs

    def detect_speech_interference(self, signal):
        """
        Detect segments with speech interference (not DDK syllables)
        Speech has different spectral characteristics than DDK repetitions
        """
        # Compute spectrogram
        nperseg = min(1024, len(signal) // 10)
        noverlap = nperseg // 2
        freqs, times, Sxx = spectrogram(signal, fs=self.fs, nperseg=nperseg, noverlap=noverlap)

        # Speech typically has energy in lower frequencies (formants)
        low_freq_mask = (freqs >= 100) & (freqs <= 500)
        mid_freq_mask = (freqs >= 500) & (freqs <= 2000)
        high_freq_mask = (freqs >= 2000) & (freqs <= 8000)

        low_energy = np.sum(Sxx[low_freq_mask, :], axis=0)
        mid_energy = np.sum(Sxx[mid_freq_mask, :], axis=0)
        high_energy = np.sum(Sxx[high_freq_mask, :], axis=0)

        # Speech has more variation in formant structure
        # DDK is more rhythmic and repetitive
        total_energy = low_energy + mid_energy + high_energy

        # Detect long sustained segments (likely speech, not DDK)
        window_size = max(1, len(times) // 20)
        rolling_std = np.array([np.std(total_energy[max(0, i - window_size):i + window_size])
                                for i in range(len(total_energy))])

        # High variation = likely speech
        speech_threshold = np.percentile(rolling_std, 70)
        potential_speech = rolling_std > speech_threshold

        # Convert time indices to sample indices
        speech_regions = []
        in_speech = False
        start_time = 0

        for i, is_speech in enumerate(potential_speech):
            if is_speech and not in_speech:
                start_time = times[i]
                in_speech = True
            elif not is_speech and in_speech:
                end_time = times[i]
                if end_time - start_time > 0.3:  # Only mark if > 300ms
                    speech_regions.append((int(start_time * self.fs),
                                           int(end_time * self.fs)))
                in_speech = False

        return speech_regions

    def detect_background_noise(self, signal):
        """
        Detect segments with only background noise (no DDK activity)
        """
        # Calculate short-term energy
        frame_length = int(0.02 * self.fs)  # 20ms frames
        hop_length = frame_length // 2

        energy = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy = np.array(energy)

        # Noise floor is in the lowest 15% of energy
        noise_threshold = np.percentile(energy, 15)

        # Find continuous low-energy regions
        is_noise = energy < noise_threshold * 1.5

        noise_regions = []
        in_noise = False
        start_idx = 0

        for i, is_low in enumerate(is_noise):
            if is_low and not in_noise:
                start_idx = i
                in_noise = True
            elif not is_low and in_noise:
                # Convert frame indices to sample indices
                start_sample = start_idx * hop_length
                end_sample = i * hop_length
                if end_sample - start_sample > 0.1 * self.fs:  # > 100ms
                    noise_regions.append((start_sample, end_sample))
                in_noise = False

        return noise_regions, noise_threshold * 1.5

    def detect_clipping(self, signal):
        """Detect clipping artifacts"""
        # Normalize to -1 to 1 range
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            norm_signal = signal / max_val
        else:
            return []

        # Find samples at or near maximum
        clipping_threshold = 0.98
        clipped = np.abs(norm_signal) > clipping_threshold

        # Find regions of consecutive clipping
        clipping_regions = []
        in_clip = False
        start_idx = 0

        for i, is_clipped in enumerate(clipped):
            if is_clipped and not in_clip:
                start_idx = i
                in_clip = True
            elif not is_clipped and in_clip:
                clipping_regions.append((start_idx, i))
                in_clip = False

        return clipping_regions

    def apply_spectral_gating(self, signal, noise_profile_region=None):
        """
        Apply spectral gating to reduce noise
        Uses noise profile from quiet regions
        """
        # If no noise profile region specified, use first 0.5 seconds
        if noise_profile_region is None:
            noise_profile_region = (0, min(int(0.5 * self.fs), len(signal) // 4))

        # Get noise profile
        noise_sample = signal[noise_profile_region[0]:noise_profile_region[1]]

        # Compute STFT
        nperseg = 512
        noverlap = nperseg // 2

        freqs, times, Sxx = spectrogram(signal, fs=self.fs, nperseg=nperseg,
                                        noverlap=noverlap, mode='complex')

        # Estimate noise spectrum from noise sample
        _, _, noise_Sxx = spectrogram(noise_sample, fs=self.fs, nperseg=nperseg,
                                      noverlap=noverlap)
        noise_spectrum = np.median(np.abs(noise_Sxx), axis=1, keepdims=True)

        # Apply gating
        magnitude = np.abs(Sxx)
        phase = np.angle(Sxx)

        # Compute SNR for each time-frequency bin
        snr = magnitude / (noise_spectrum + 1e-10)

        # Create soft mask (smoother than hard gating)
        threshold = 1.5
        mask = np.minimum(1.0, np.maximum(0.0, (snr - 1) / (threshold - 1)))

        # Apply mask
        gated_Sxx = magnitude * mask * np.exp(1j * phase)

        # Reconstruct signal
        from scipy.signal import istft
        _, cleaned_signal = istft(gated_Sxx, fs=self.fs, nperseg=nperseg,
                                  noverlap=noverlap)

        # Ensure same length as input
        if len(cleaned_signal) < len(signal):
            cleaned_signal = np.pad(cleaned_signal,
                                    (0, len(signal) - len(cleaned_signal)))
        else:
            cleaned_signal = cleaned_signal[:len(signal)]

        return cleaned_signal

    def apply_highpass_filter(self, signal, cutoff=80):
        """Remove low-frequency rumble"""
        nyquist = self.fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='high')
        filtered = filtfilt(b, a, signal)
        return filtered

    def remove_clicks(self, signal, threshold=3.0):
        """Remove click artifacts using median filtering"""
        # Detect outliers
        median_sig = median_filter(signal, size=5)
        diff = np.abs(signal - median_sig)
        mad = np.median(diff)

        # Mark clicks
        is_click = diff > threshold * mad

        # Replace clicks with median value
        cleaned = signal.copy()
        cleaned[is_click] = median_sig[is_click]

        return cleaned


class DDKSegmenter:

    def __init__(self, target_fs=20000, enable_noise_removal=True, pattern_type='pataka'):
        self.target_fs = target_fs
        self.trim_offset = 0
        self.enable_noise_removal = enable_noise_removal
        self.noise_detector = None
        self.pattern_type = pattern_type

    def step1_preprocessing(self, audio_path):
        """Pre-processing: resample, normalize, denoise"""
        print("Step 1: Pre-processing")
        print("-" * 40)

        # Robust file reading
        try:
            original_fs, signal = wavfile.read(audio_path)
            print(f"  Read with scipy.io.wavfile")
        except (ValueError, Exception) as e:
            print(f"  scipy failed: {e}")
            try:
                import soundfile as sf
                signal, original_fs = sf.read(audio_path, dtype='float32')
                signal = signal * 32768
                print(f"  Read with soundfile")
            except ImportError:
                import wave
                with wave.open(audio_path, 'rb') as wav:
                    original_fs = wav.getframerate()
                    n_frames = wav.getnframes()
                    raw_data = wav.readframes(n_frames)
                    signal = np.frombuffer(raw_data, dtype=np.int16)
                print(f"  Read with wave module")

        print(f"  Original sampling rate: {original_fs} Hz")
        self.original_duration = len(signal) / original_fs

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
            print("  Converted to mono")

        signal = signal.astype(float)

        if original_fs != self.target_fs:
            num_samples = int(len(signal) * self.target_fs / original_fs)
            signal = resample(signal, num_samples)
            fs = self.target_fs
            print(f"  Resampled to: {fs} Hz")
        else:
            fs = original_fs

        # Initialize noise detector
        self.noise_detector = NoiseDetector(fs)

        if self.enable_noise_removal:
            print("\n  Noise Detection & Removal:")

            # Detect various noise types
            speech_regions = self.noise_detector.detect_speech_interference(signal)
            if speech_regions:
                print(f"    ⚠ Detected {len(speech_regions)} potential speech interference region(s)")
                for i, (start, end) in enumerate(speech_regions):
                    print(f"      Region {i + 1}: {start / fs:.2f}s - {end / fs:.2f}s")

            noise_regions, noise_threshold = self.noise_detector.detect_background_noise(signal)
            if noise_regions:
                print(f"    Found {len(noise_regions)} low-energy (noise) region(s)")

            clipping_regions = self.noise_detector.detect_clipping(signal)
            if clipping_regions:
                print(f"    ⚠ Detected {len(clipping_regions)} clipping region(s)")

            # Apply noise reduction
            print("    Applying noise reduction...")

            # High-pass filter to remove rumble
            signal = self.noise_detector.apply_highpass_filter(signal, cutoff=80)
            print("      ✓ High-pass filter (80 Hz)")

            # Remove clicks
            signal = self.noise_detector.remove_clicks(signal)
            print("      ✓ Click removal")

            # Spectral gating if we have noise regions
            if noise_regions:
                signal = self.noise_detector.apply_spectral_gating(signal, noise_regions[0])
                print("      ✓ Spectral noise gating")

        # Normalize
        signal = signal - np.mean(signal)
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 0:
            signal = signal / max_amplitude
        print(f"  Normalized to [-1, 1]")

        duration = len(signal) / fs
        print(f"  Final duration: {duration:.2f} seconds")

        return signal, fs

    def detect_burst_from_spectrogram(self, signal, search_start, search_end, fs, consonant_type='t'):
        """
        Detect burst onset by analyzing spectrogram for sudden broadband energy.
        """
        if search_end <= search_start:
            return search_start

        region = signal[search_start:search_end]
        if len(region) < 512:
            return search_start

        if consonant_type == 'p':
            nperseg = min(256, len(region) // 4)
        else:
            nperseg = min(512, len(region) // 4)
        noverlap = nperseg // 2

        freqs, times, Sxx = spectrogram(region, fs=fs, nperseg=nperseg, noverlap=noverlap)

        if consonant_type == 't':
            freq_mask = (freqs >= 2000) & (freqs <= 8000)
        elif consonant_type == 'p':
            freq_mask = (freqs >= 500) & (freqs <= 5000)
        else:  # 'k'
            freq_mask = (freqs >= 2000) & (freqs <= 6000)

        burst_energy = np.sum(Sxx[freq_mask, :], axis=0)

        if len(burst_energy) < 3:
            return search_start

        onset_strength = np.diff(burst_energy)
        onset_strength = np.concatenate([[0], onset_strength])

        if consonant_type == 'p':
            threshold = np.percentile(onset_strength, 65)
            energy_threshold = np.percentile(burst_energy, 20)
        else:
            threshold = np.percentile(onset_strength, 75)
            energy_threshold = np.percentile(burst_energy, 30)

        for i in range(len(onset_strength)):
            if onset_strength[i] > threshold and burst_energy[i] > energy_threshold:
                time_offset = times[i]
                burst_sample = search_start + int(time_offset * fs)
                return burst_sample

        if consonant_type == 'p':
            for i in range(len(burst_energy)):
                if burst_energy[i] > np.percentile(burst_energy, 25):
                    time_offset = times[i]
                    burst_sample = search_start + int(time_offset * fs)
                    return burst_sample
        else:
            median_burst = np.median(burst_energy)
            for i in range(len(burst_energy)):
                if burst_energy[i] > median_burst * 1.5:
                    time_offset = times[i]
                    burst_sample = search_start + int(time_offset * fs)
                    return burst_sample

        return search_start

    def step2_smooth_signal(self, signal, fs):
        """Apply moving average smoothing"""
        print("\nStep 2: Signal smoothing")
        print("-" * 40)

        window_length = int(fs / 25)
        print(f"  Window length: {window_length} samples")

        kernel = np.ones(window_length) / window_length
        abs_signal = np.abs(signal)
        smoothed = np.convolve(abs_signal, kernel, mode='same')

        second_window = int(window_length / 2)
        kernel2 = np.ones(second_window) / second_window
        smoothed = np.convolve(smoothed, kernel2, mode='same')

        print(f"  Two-pass smoothing applied")
        return smoothed

    def step3_identify_peaks(self, smoothed_signal, fs):
        """Identify main vowel peaks only"""
        print("\nStep 3: Peak identification")
        print("-" * 40)

        min_distance = int(0.12 * fs)

        if np.any(smoothed_signal > 0):
            height_threshold = np.percentile(smoothed_signal[smoothed_signal > 0], 65)
            prominence = np.std(smoothed_signal[smoothed_signal > 0]) * 0.5
        else:
            height_threshold = 0
            prominence = 0

        peaks, properties = find_peaks(
            smoothed_signal,
            height=height_threshold,
            distance=min_distance,
            prominence=prominence if prominence > 0 else None,
            width=int(0.015 * fs)
        )

        print(f"  Found {len(peaks)} peaks")

        # Limit based on pattern type
        if self.pattern_type in ['papapa', 'tatata', 'kakaka']:
            max_syllables = 10  # AMR: only 10 syllables
        else:
            max_syllables = 30  # DDK: 10 repetitions = 30 syllables

        if len(peaks) > max_syllables:
            print(f"  Limiting to {max_syllables} syllables")
            peaks = peaks[:max_syllables]

        if self.pattern_type in ['papapa', 'tatata', 'kakaka']:
            print(f"  Detected {len(peaks)} AMR syllables")
        elif len(peaks) > 0 and len(peaks) % 3 == 0:
            n_cycles = len(peaks) // 3
            print(f"  Detected {n_cycles} complete pa-ta-ka cycles")

        return peaks

    def step4_segment_syllables(self, signal, peaks, fs):
        """Segment syllables with spectrogram-based burst detection"""
        print("\nStep 4: Syllable segmentation")
        print("-" * 40)

        if len(peaks) == 0:
            return []

        segments = []

        window_ms = 10
        window_samples = int(window_ms * fs / 1000)
        hop_samples = window_samples // 2

        energy = []
        for i in range(0, len(signal) - window_samples, hop_samples):
            window = signal[i:i + window_samples]
            rms = np.sqrt(np.mean(window ** 2))
            energy.append(rms)
        energy = np.array(energy)

        median_energy = np.median(energy[energy > 0]) if np.any(energy > 0) else 0

        # Determine consonant pattern based on pattern_type
        if self.pattern_type == 'papapa':
            consonant_sequence = ['p'] * len(peaks)
        elif self.pattern_type == 'tatata':
            consonant_sequence = ['t'] * len(peaks)
        elif self.pattern_type == 'kakaka':
            consonant_sequence = ['k'] * len(peaks)
        else:  # pataka or badaga
            consonant_sequence = [['p', 't', 'k'][i % 3] for i in range(len(peaks))]

        for i in range(len(peaks)):
            peak_sample = peaks[i]
            consonant_type = consonant_sequence[i]

            # For AMR, rep number is just the syllable number
            if self.pattern_type in ['papapa', 'tatata', 'kakaka']:
                rep = i + 1
            else:
                rep = (i // 3) + 1

            if i == 0:
                search_start = max(0, peak_sample - int(0.15 * fs))
                search_end = peak_sample - int(0.02 * fs)
            else:
                prev_end_sample = segments[-1]['end_sample']
                search_start = int(prev_end_sample + 0.005 * fs)
                search_end = peak_sample - int(0.02 * fs)

            burst_sample = self.detect_burst_from_spectrogram(
                signal, search_start, search_end, fs, consonant_type
            )

            if burst_sample == search_start:
                threshold = np.percentile(energy, 25)
                for j in range(search_start, min(search_end, len(signal)), hop_samples):
                    idx = j // hop_samples
                    if idx < len(energy) and energy[idx] > threshold:
                        burst_sample = j
                        break

            vot_min = int(0.015 * fs)
            vot_max = int(0.080 * fs)

            if consonant_type == 't':
                vot_max = int(0.060 * fs)

            vot_search_start = burst_sample + vot_min
            vot_search_end = min(burst_sample + vot_max, peak_sample - int(0.01 * fs))

            vowel_onset_sample = burst_sample + int(0.025 * fs)

            for j in range(vot_search_start, vot_search_end, hop_samples):
                idx = j // hop_samples
                if idx + 4 < len(energy):
                    window = energy[idx:idx + 4]

                    stability_threshold = 0.35 if consonant_type == 't' else 0.30
                    energy_threshold = 1.1 if consonant_type == 't' else 1.2

                    if (np.mean(window) > median_energy * energy_threshold and
                            np.std(window) / (np.mean(window) + 1e-10) < stability_threshold):
                        vowel_onset_sample = j
                        break

            if i == len(peaks) - 1:
                search_start = peak_sample + int(0.02 * fs)
                search_end = min(len(signal), peak_sample + int(0.20 * fs))

                end_sample = search_end
                peak_idx = peak_sample // hop_samples
                if peak_idx < len(energy):
                    drop_threshold = energy[peak_idx] * 0.3

                    for j in range(search_start, search_end, hop_samples):
                        idx = j // hop_samples
                        if idx < len(energy) and energy[idx] < drop_threshold:
                            end_sample = j
                            break
            else:
                next_peak = peaks[i + 1]
                search_start = peak_sample + int(0.02 * fs)
                search_end = next_peak - int(0.05 * fs)

                if search_end > search_start:
                    min_energy = float('inf')
                    min_pos = (peak_sample + next_peak) // 2

                    for j in range(search_start, search_end, hop_samples):
                        idx = j // hop_samples
                        if idx < len(energy) and energy[idx] < min_energy:
                            min_energy = energy[idx]
                            min_pos = j

                    end_sample = min_pos
                else:
                    end_sample = (peak_sample + next_peak) // 2

            segment_info = {
                'burst_sample': burst_sample,
                'vowel_onset_sample': vowel_onset_sample,
                'end_sample': end_sample,
                'start': burst_sample / fs,
                'vowel_onset': vowel_onset_sample / fs,
                'end': end_sample / fs,
                'vot_duration': (vowel_onset_sample - burst_sample) / fs * 1000,
                'consonant': consonant_type
            }

            segments.append(segment_info)

            print(f"  Seg {i + 1} ({consonant_type}): Burst={segment_info['start']:.3f}s, " +
                  f"Vowel={segment_info['vowel_onset']:.3f}s, " +
                  f"End={segment_info['end']:.3f}s (VOT={segment_info['vot_duration']:.1f}ms)")

        return segments

    def create_textgrid(self, segments, duration, pattern_type='pataka'):
        """Create TextGrid with single segment tier containing VOT and vowel portions"""
        textgrid_duration = self.original_duration
        time_offset = self.trim_offset

        textgrid = tgt.TextGrid()
        segment_tier = tgt.IntervalTier(0, textgrid_duration, 'segment')

        # Determine if this is AMR or SMR
        is_amr = pattern_type in ['papapa', 'tatata', 'kakaka']

        # Determine consonant pattern
        if pattern_type == 'papapa':
            consonants = ['p'] * len(segments)
        elif pattern_type == 'tatata':
            consonants = ['t'] * len(segments)
        elif pattern_type == 'kakaka':
            consonants = ['k'] * len(segments)
        else:  # pataka
            consonants = [['p', 't', 'k'][i % 3] for i in range(len(segments))]

        # SEGMENT TIER - VOT followed directly by vowel portions
        last_end = 0
        for i, seg in enumerate(segments):
            c = consonants[i]
        
            # For AMR, rep number is just the syllable number
            # For SMR, rep number is the cycle number
            if is_amr:
                rep = i + 1
            else:
                rep = (i // 3) + 1

            burst_t = seg['start'] + time_offset
            vowel_t = seg['vowel_onset'] + time_offset
            end_t = seg['end'] + time_offset

            # Ensure times are valid and in sequence
            burst_t = max(burst_t, last_end)
            vowel_t = max(vowel_t, burst_t + 0.001)
            end_t = max(end_t, vowel_t + 0.001)

            # Gap before burst (if any)
            if burst_t > last_end + 0.001:
                segment_tier.add_interval(tgt.Interval(last_end, burst_t, ""))

            # VOT interval
            segment_tier.add_interval(tgt.Interval(burst_t, vowel_t, f"{c}{rep:02d}VOT"))

            # Vowel interval directly after VOT (no gap)
            segment_tier.add_interval(tgt.Interval(vowel_t, end_t, f"{c}uh{rep:02d}"))

            last_end = end_t

        # Fill to end of file
        if last_end < textgrid_duration:
            segment_tier.add_interval(tgt.Interval(last_end, textgrid_duration, ""))

        # Add only the segment tier to the textgrid
        textgrid.add_tier(segment_tier)

        return textgrid

    def visualize_segmentation(self, signal, fs, smoothed, peaks, segments, output_path=None):
        """Visualization with burst, VOT, and vowel markers"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        time = np.arange(len(signal)) / fs
        time_smoothed = np.arange(len(smoothed)) / fs

        ax1.plot(time, signal, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal with Burst (solid), VOT end (dashed), Syllable end (dotted)')
        ax1.grid(True, alpha=0.3)

        colors = ['red', 'blue', 'green']
        for i, seg in enumerate(segments):
            color = colors[i % 3]

            ax1.axvspan(seg['start'], seg['end'], alpha=0.1, color=color)
            ax1.axvline(seg['start'], color=color, linestyle='-', linewidth=1.5,
                        label='Burst' if i == 0 else '')
            ax1.axvline(seg['vowel_onset'], color=color, linestyle='--', linewidth=1.0,
                        label='VOT end' if i == 0 else '')
            ax1.axvline(seg['end'], color=color, linestyle=':', linewidth=0.8, alpha=0.5,
                        label='Syl end' if i == 0 else '')

        if segments:
            ax1.legend(loc='upper right')

        ax2.plot(time_smoothed, smoothed, 'g-', linewidth=1)
        ax2.plot(peaks / fs, smoothed[peaks], 'ro', markersize=8, label='Vowel peaks')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Smoothed Amplitude')
        ax2.set_title('Smoothed Signal with Detected Peaks')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        for peak in peaks:
            ax2.axvline(peak / fs, color='red', linestyle=':', alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved visualization to: {output_path}")

        plt.show()
        return fig

    def process(self, audio_path, visualize=False):
        """Process audio through all steps"""
        print("=" * 60)
        print(f"AMR/SMR Segmentation with Noise Detection ({self.pattern_type})")
        print("=" * 60)

        signal, fs = self.step1_preprocessing(audio_path)
        smoothed = self.step2_smooth_signal(signal, fs)
        peaks = self.step3_identify_peaks(smoothed, fs)
        segments = self.step4_segment_syllables(signal, peaks, fs)

        if visualize:
            vis_path = Path(audio_path).with_suffix('.png')
            self.visualize_segmentation(signal, fs, smoothed, peaks, segments, vis_path)

        duration = len(signal) / fs
        textgrid = self.create_textgrid(segments, duration, pattern_type=self.pattern_type)

        output_path = Path(audio_path).with_suffix('.TextGrid')
        tgt.write_to_file(textgrid, str(output_path), format='long')
        print(f"\nSaved TextGrid to: {output_path}")

        return {
            'signal': signal,
            'fs': fs,
            'smoothed': smoothed,
            'peaks': peaks,
            'segments': segments,
            'textgrid': textgrid
        }


def detect_pattern_from_path(filepath):
    """
    Detect pattern type by checking for 'amr' or 'smr' in directory structure.
    For AMR folders, determines specific pattern from filename.
    For SMR folders, returns 'pataka'.
    Falls back to filename detection if no folder hints found.
    
    Args:
        filepath: Path object or string to the audio file
        
    Returns:
        Detected pattern string or None if not found
    """
    filepath = Path(filepath)
    filename = filepath.stem
    
    # Check directory structure (up to 3 levels up)
    for parent in [filepath.parent, filepath.parent.parent, filepath.parent.parent.parent]:
        parent_name = parent.name.lower()
        
        # Check for AMR folder
        if 'amr' in parent_name:
            # Determine which AMR pattern from filename
            filename_lower = filename.lower()
            filename_normalized = filename_lower.replace('_', '').replace('-', '').replace(' ', '')
            
            # Check for specific AMR patterns
            if any(p in filename_normalized for p in ['pa', 'puh', 'papapa']):
                return 'papapa'
            elif any(p in filename_normalized for p in ['ta', 'tuh', 'tatata']):
                return 'tatata'
            elif any(p in filename_normalized for p in ['ka', 'kuh', 'kakaka']):
                return 'kakaka'
            else:
                # Default to papapa if in AMR folder but can't determine which
                print(f"  ⚠ Found AMR folder but couldn't determine specific pattern from filename")
                print(f"    Defaulting to 'papapa' - use --pattern to specify if different")
                return 'papapa'
        
        # Check for SMR folder
        elif 'smr' in parent_name:
            return 'pataka'
    
    # If no directory hints, fall back to filename detection
    return detect_pattern_from_filename(filename)


def detect_pattern_from_filename(filename):
    """
    Fallback filename-based detection when no directory structure hints.
    """
    filename_lower = filename.lower()
    filename_normalized = filename_lower.replace('_', '').replace('-', '').replace(' ', '')
    
    # Lab-specific naming convention
    if 'rhythmpa' in filename_normalized or filename_normalized.endswith('pa'):
        return 'papapa'
    elif 'rhythmta' in filename_normalized or filename_normalized.endswith('ta'):
        return 'tatata'
    elif 'rhythmka' in filename_normalized or filename_normalized.endswith('ka'):
        return 'kakaka'
    elif any(p in filename_normalized for p in ['rhythmptk', 'rhythmddk', 'rhythmsmr', 'pataka', 'ptk', 'ddk', 'smr']):
        return 'pataka'
    
    # Check for repeated syllables
    if any(pattern in filename_normalized for pattern in ['papapa', 'puhpuhpuh']):
        return 'papapa'
    elif any(pattern in filename_normalized for pattern in ['tatata', 'tuhtuhtu']):
        return 'tatata'
    elif any(pattern in filename_normalized for pattern in ['kakaka', 'kuhkuhkuh']):
        return 'kakaka'
    
    # Single syllable indicators
    if any(s in filename_normalized for s in ['pa', 'puh']) and 'pataka' not in filename_normalized:
        return 'papapa'
    elif any(s in filename_normalized for s in ['ta', 'tuh']) and 'pataka' not in filename_normalized:
        return 'tatata'
    elif any(s in filename_normalized for s in ['ka', 'kuh']) and 'pataka' not in filename_normalized:
        return 'kakaka'
    
    return None


def process_batch(input_path, visualize=False, target_fs=20000,
                  enable_denoise=True, pattern=None, skip_existing=True):
    """
    Process multiple audio files in batch mode

    Args:
        input_path: Path to directory containing WAV files or path to single WAV file
        visualize: Whether to create visualizations
        target_fs: Target sampling frequency
        enable_denoise: Enable noise removal
        pattern: Pattern type (pataka, papapa, tatata, kakaka, badaga).
                If None, will attempt to detect from filename.
        skip_existing: If True, skip files that already have TextGrid files
    """
    input_path = Path(input_path)

    # Determine if input is a file or directory
    if input_path.is_file():
        if input_path.suffix.lower() != '.wav':
            print(f"Error: {input_path} is not a WAV file")
            return
        audio_files = [input_path]
    elif input_path.is_dir():
        # Find all WAV files in directory
        audio_files = sorted(input_path.glob('*.wav'))
        if not audio_files:
            print(f"No WAV files found in {input_path}")
            return
    else:
        print(f"Error: {input_path} does not exist")
        return

    # Filter out files that already have TextGrids if skip_existing is True
    if skip_existing:
        files_to_process = []
        skipped_files = []
        for audio_file in audio_files:
            textgrid_path = audio_file.with_suffix('.TextGrid')
            if textgrid_path.exists():
                skipped_files.append(audio_file)
            else:
                files_to_process.append(audio_file)

        if skipped_files:
            print(f"\n{'=' * 60}")
            print(f"Skipping {len(skipped_files)} file(s) with existing TextGrids:")
            for f in skipped_files:
                print(f"  ✓ {f.name} (already segmented)")
            print(f"{'=' * 60}\n")

        audio_files = files_to_process

    if not audio_files:
        print("No files to process (all files already have TextGrids)")
        return

    print(f"\n{'=' * 60}")
    print(f"BATCH PROCESSING: {len(audio_files)} file(s)")
    print(f"{'=' * 60}\n")

    successful = []
    failed = []

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'#' * 60}")
        print(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
        print(f"{'#' * 60}\n")

        # Determine pattern for this file
        file_pattern = pattern  # Use command-line pattern if specified

        if file_pattern is None:
            # Try to detect from filename
            detected_pattern = detect_pattern_from_path(audio_file)
            if detected_pattern:
                file_pattern = detected_pattern
                print(f"  → Auto-detected pattern: {file_pattern}")
            else:
                # Default to pataka if can't detect
                file_pattern = 'pataka'
                print(f"  ⚠ Could not detect pattern from filename, using default: {file_pattern}")

        try:
            # Create a new segmenter for each file with its specific pattern
            segmenter = DDKSegmenter(
                target_fs=target_fs,
                enable_noise_removal=enable_denoise,
                pattern_type=file_pattern
            )

            segmenter.process(str(audio_file), visualize=visualize)
            successful.append((audio_file, file_pattern))
            print(f"\n✓ Successfully processed: {audio_file.name} (pattern: {file_pattern})")
        except Exception as e:
            failed.append((audio_file, str(e)))
            print(f"\n✗ Failed to process {audio_file.name}: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successfully processed: {len(successful)}/{len(audio_files)}")
    if successful:
        print("\nProcessed files by pattern:")
        pattern_counts = {}
        for audio_file, file_pattern in successful:
            pattern_counts[file_pattern] = pattern_counts.get(file_pattern, 0) + 1
        for pat, count in sorted(pattern_counts.items()):
            print(f"  {pat}: {count} file(s)")

    if skip_existing and skipped_files:
        print(f"\nSkipped (existing):    {len(skipped_files)}")
    if failed:
        print(f"\nFailed:                {len(failed)}")
        print("\nFailed files:")
        for audio_file, error in failed:
            print(f"  ✗ {audio_file.name}: {error}")
    print(f"{'=' * 60}\n")

def main():
    parser = argparse.ArgumentParser(
        description='SMR/AMR Segmentation with Noise Detection and Batch Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python script.py audio.wav

  # Process all WAV files in a directory
  python script.py /path/to/audio/directory/

  # Process with visualization
  python script.py audio.wav --visualize

  # Process directory, overwriting existing TextGrids
  python script.py /path/to/audio/ --no-skip-existing

  # Process AMR pattern
  python script.py audio.wav --pattern papapa
        """
    )

    parser.add_argument('input_path',
                        help='Path to WAV file or directory containing WAV files')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--target-fs', type=int, default=20000,
                        help='Target sampling frequency (default: 20000)')
    parser.add_argument('--no-denoise', action='store_true',
                        help='Disable noise removal')
    parser.add_argument('--pattern', type=str, default=None,
                        choices=['pataka', 'papapa', 'tatata', 'kakaka', 'badaga'],
                        help='Pattern type. If not specified, will auto-detect from filename.')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Process all files even if TextGrids already exist (will overwrite)')

    args = parser.parse_args()

    process_batch(
        input_path=args.input_path,
        visualize=args.visualize,
        target_fs=args.target_fs,
        enable_denoise=not args.no_denoise,
        pattern=args.pattern,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()
