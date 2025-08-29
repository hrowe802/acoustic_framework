#!/usr/bin/env python
# coding: utf-8

# In[33]:


#!/usr/bin/env python3
"""
Complete Acoustic Analysis Pipeline for AMR/SMR Data
Fixed version with proper MATLAB-compatible formant processing
"""

import os
import parselmouth
import pandas as pd
import numpy as np
import csv
import glob
from parselmouth.praat import call
from textgrid import TextGrid
import re
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================
TASK_TYPE = "SMR"  # Change to "AMR" for AMR data
MAX_FILES = 5      # Set to None to process all files

# Set directories
if TASK_TYPE == "SMR":
    project_dir = "drive-download-20250825T142829Z-1-001/"
else:
    project_dir = "."

output_dir = "TRIAL_ACOUSTIC_AMR_3/" if TASK_TYPE == "AMR" else "TRIAL_ACOUSTIC_SMR_4/"
os.makedirs(output_dir, exist_ok=True)

print(f"Processing {TASK_TYPE} data")
print(f"Input directory: {project_dir}")
print(f"Output directory: {output_dir}")
if MAX_FILES:
    print(f"Limited to first {MAX_FILES} files")
print("=" * 60)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_amr_label(label):
    """Parse AMR labels like 'kuh01' to extract consonant and rep number"""
    label = label.lower().strip()
    
    # Extract consonant
    if 'kuh' in label or 'k' in label:
        consonant = 'k'
    elif 'puh' in label or 'p' in label:
        consonant = 'p'
    elif 'tuh' in label or 't' in label:
        consonant = 't'
    else:
        return None, None
    
    # Extract repetition number
    rep_match = re.search(r'(\d+)', label)
    if rep_match:
        rep_num = int(rep_match.group(1))
    else:
        rep_num = 1
    
    return consonant, rep_num

# ============================================================================
# 1. CPP EXTRACTION
# ============================================================================

def calculate_cpp(snd, onset, offset):
    sound_part = snd.extract_part(from_time=onset, to_time=offset, 
                                  window_shape=parselmouth.WindowShape.HAMMING, 
                                  relative_width=1.988)
    parselmouth.praat.call(sound_part, "To Formant (burg)", 0, 5, 5000, 0.0025, 50)
    
    power_cepstrogram = parselmouth.praat.call(sound_part, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    parselmouth.praat.call(power_cepstrogram, "Smooth", 0.02, 0.0005)
    
    cpps = parselmouth.praat.call(power_cepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 
                                  60, 330, 0.05, "Parabolic", 0.001, 0, 
                                  "Exponential decay", "Robust")
    return cpps

def process_file_cpp(textgrid_path, snd, participant_name):
    output = []
    tg = parselmouth.praat.call("Read from file", textgrid_path)
    
    if TASK_TYPE == "AMR":
        num_intervals = parselmouth.praat.call(tg, "Get number of intervals", 3)
        
        for i in range(1, num_intervals + 1):
            label = parselmouth.praat.call(tg, "Get label of interval", 3, i)
            if label and not label.isspace():
                onset = parselmouth.praat.call(tg, "Get start point", 3, i)
                offset = parselmouth.praat.call(tg, "Get end point", 3, i)
                cpps = calculate_cpp(snd, onset, offset)
                
                consonant, rep = parse_amr_label(label)
                if consonant:
                    task = f"{consonant}uh{rep}"
                    output.append([participant_name, task, rep, cpps])
    
    else:  # SMR
        num_intervals = parselmouth.praat.call(tg, "Get number of intervals", 3)
        for i in range(1, num_intervals + 1):
            label = parselmouth.praat.call(tg, "Get label of interval", 3, i)
            if label.startswith("puhtuhkuh"):
                onset = parselmouth.praat.call(tg, "Get start point", 3, i)
                offset = parselmouth.praat.call(tg, "Get end point", 3, i)
                cpps = calculate_cpp(snd, onset, offset)

                rep_match = re.search(r'(\d+)', label)
                rep = int(rep_match.group(1)) if rep_match else 1
                
                for consonant in ['p', 't', 'k']:
                    task = f"{consonant}uh{rep}"
                    output.append([participant_name, task, rep, cpps])
    
    return output

def main_cpp():
    output_rows = [["Participant", "Task", "Rep", "CPPS"]]
    
    files_processed = 0
    for filename in os.listdir(project_dir):
        if not filename.endswith(".TextGrid"):
            continue
        
        if MAX_FILES and files_processed >= MAX_FILES:
            print(f"Reached max file limit ({MAX_FILES})")
            break

        textgrid_path = os.path.join(project_dir, filename)
        wav_path = textgrid_path.replace(".TextGrid", ".wav")
        participant_name = filename.replace(".TextGrid", "")

        if not os.path.exists(wav_path):
            print(f"Missing wav file for {filename}")
            continue

        try:
            snd = parselmouth.Sound(wav_path)
            output_rows.extend(process_file_cpp(textgrid_path, snd, participant_name))
            files_processed += 1
            print(f"Processed CPP for {filename} ({files_processed}/{MAX_FILES if MAX_FILES else 'all'})")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    intermediate_csv = os.path.join(output_dir, "Cepstrum_Data_For_R.csv")
    with open(intermediate_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(output_rows)

    final_csv = os.path.join(output_dir, "Cepstrum_Data_For_Spreadsheets.csv")
    pd.read_csv(intermediate_csv).to_csv(final_csv, index=False)

    print(f"CPP processing complete. Data saved to {final_csv}")
    return pd.read_csv(final_csv)

# ============================================================================
# 2. FORMANT EXTRACTION (FIXED TO MATCH MATLAB)
# ============================================================================

def main_formant():
    """Extract raw formant data with PI's suggested settings"""
    print("Starting formant extraction with MAX_FORMANTS = 5")
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
        print(f"Limited to {MAX_FILES} files")
    
    results = []
    
    # Test with 5 formants first (or change to 4)
    MAX_FORMANTS = 4  # Try 4 or 5 as your PI suggested

    for textgrid_file in textgrid_files:
        filename = os.path.basename(textgrid_file).replace('.TextGrid', '')
        textgrid_path = os.path.join(project_dir, filename + ".TextGrid")
        sound_path = os.path.join(project_dir, filename + ".wav")

        if not os.path.exists(sound_path):
            print(f"Sound file for {filename} not found. Skipping.")
            continue

        try:
            tg = parselmouth.Data.read(textgrid_path)
            sound = parselmouth.Sound(sound_path)
            
            number_of_intervals = call(tg, "Get number of intervals", 1)
            
            # Create formant with PI's settings
            formant = call(sound, "To Formant (burg)...", 
                          0,              # time step
                          MAX_FORMANTS,   # 4 or 5
                          5500,          # max frequency
                          0.025,         # window length  
                          50)            # pre-emphasis

            for interval_index in range(1, number_of_intervals + 1):
                phoneme = call(tg, "Get label of interval", 1, interval_index).strip()

                if phoneme != "":
                    start_time = call(tg, "Get start point", 1, interval_index)
                    end_time = call(tg, "Get end point", 1, interval_index)
                    duration = end_time - start_time

                    # Skip VOT intervals
                    if 'vot' in phoneme.lower():
                        continue
                    
                    # Parse task name
                    phoneme_lower = phoneme.lower()
                    
                    if TASK_TYPE == "AMR":
                        consonant, rep = parse_amr_label(phoneme)
                        if not consonant:
                            continue
                        task_name = f"{consonant}uh{rep}"
                    else:  # SMR
                        if 'puh' in phoneme_lower:
                            consonant = 'p'
                        elif 'tuh' in phoneme_lower:
                            consonant = 't'
                        elif 'kuh' in phoneme_lower:
                            consonant = 'k'
                        else:
                            continue
                        
                        rep_match = re.search(r'(\d+)', phoneme_lower)
                        rep = int(rep_match.group(1)) if rep_match else 1
                        task_name = f"{consonant}uh{rep}"

                    # Sample at 0.0025 intervals
                    frame_count = int(duration / 0.0025)

                    for frame in range(frame_count):
                        frame_time = start_time + (frame * 0.0025)
                        f1 = call(formant, "Get value at time", 1, frame_time, "Hertz", "Linear")
                        f2 = call(formant, "Get value at time", 2, frame_time, "Hertz", "Linear")
                        results.append([filename, task_name, frame_time, f1, f2, duration])
                        
        except Exception as e:
            print(f"Error processing formants for {filename}: {e}")

    df_results = pd.DataFrame(
        results,
        columns=["Participant", "Task", "Time", "F1", "F2", "Duration"]
    )
    
    print(f"Formant extraction complete. {len(df_results)} frames extracted.")
    if len(df_results) > 0:
        print(f"Unique tasks found: {sorted(df_results['Task'].unique())}")
    
    return df_results

def process_formant_data(df_results):
    """Process formant data matching MATLAB methodology EXACTLY"""
    pd.set_option('display.float_format', lambda x: f'{x:.12g}')

    columns = [
        'Participant', 'Task', 'Vow', 'OnsetFreq_1', 'OnsetFreq_2', 'OffsetFreq_1',
        'OffsetFreq_2', 'Range_1', 'Range_2', 'Slope_1', 'Slope_2',
        'F1xF2Xcorr', 'F1xF2Corr', 'F1xF2Cov', 'Ratio_1', 'Ratio_2',
        'F1Vel', 'F1Accel', 'F1Jerk', 'F2Vel', 'F2Accel', 'F2Jerk'
    ]
    results_table = pd.DataFrame(columns=columns)

    unique_participants = df_results['Participant'].unique()

    for participant in unique_participants:
        participant_data = df_results[df_results['Participant'] == participant]
        unique_tasks = participant_data['Task'].unique()

        for task in unique_tasks:
            task_data = participant_data[participant_data['Task'] == task]
            
            # Sort by time to ensure correct order
            task_data = task_data.sort_values('Time')
            task_numbers = task_data.iloc[:, 2:6].to_numpy()

            # Skip if too few data points
            if len(task_numbers) < 2:
                continue

            # Extract positions and time
            time_vector = task_numbers[:, 0]
            F1_position = task_numbers[:, 1]
            F2_position = task_numbers[:, 2]
            
            # Calculate derivatives exactly like MATLAB
            dt = np.diff(time_vector)
            
            # Velocity
            F1_vel_array = np.diff(F1_position) / dt
            F2_vel_array = np.diff(F2_position) / dt
            
            # Acceleration
            if len(dt) > 1:
                F1_accel_array = np.diff(F1_vel_array) / dt[:-1]
                F2_accel_array = np.diff(F2_vel_array) / dt[:-1]
            else:
                F1_accel_array = np.array([np.nan])
                F2_accel_array = np.array([np.nan])
            
            # Jerk
            if len(dt) > 2:
                F1_jerk_array = np.diff(F1_accel_array) / dt[:-2]
                F2_jerk_array = np.diff(F2_accel_array) / dt[:-2]
            else:
                F1_jerk_array = np.array([np.nan])
                F2_jerk_array = np.array([np.nan])
            
            # Take nanmean
            F1_vel = np.nanmean(F1_vel_array)
            F1_accel = np.nanmean(F1_accel_array)
            F1_jerk = np.nanmean(F1_jerk_array)
            F2_vel = np.nanmean(F2_vel_array)
            F2_accel = np.nanmean(F2_accel_array)
            F2_jerk = np.nanmean(F2_jerk_array)

            # Get FULL duration
            full_duration = task_numbers[0, 3]

            # Calculate midpoint
            mid_point = round(len(task_numbers) / 2)
            if mid_point < 1:
                mid_point = 1
            
            # Use first half for onset/offset/slope
            time_mat = task_numbers[:mid_point, 0]
            data_half_mat = task_numbers[:mid_point, 1:3]

            # Calculate onset, offset, range, and slope from HALF data
            onset_freq = np.array([data_half_mat[0, 0], data_half_mat[0, 1]])
            offset_freq = np.array([data_half_mat[-1, 0], data_half_mat[-1, 1]])
            range_mat = offset_freq - onset_freq
            
            # Duration for slope (from half data)
            dur_half = time_mat[-1] - time_mat[0] if len(time_mat) > 1 else 0
            
            # Slope in kHz/s (divide by 1000)
            if dur_half > 0:
                slope_mat = range_mat / dur_half / 1000
            else:
                slope_mat = np.array([0, 0])

            # Cross-correlation on FULL data
            if len(F1_position) > 1:
                # Remove means
                F1_centered = F1_position - np.mean(F1_position)
                F2_centered = F2_position - np.mean(F2_position)
                
                # Normalized cross-correlation
                xcorr = signal.correlate(F1_centered, F2_centered, mode='full')
                norm = np.sqrt(np.sum(F1_centered**2) * np.sum(F2_centered**2))
                if norm > 0:
                    xcorr = xcorr / norm
                F1xF2_xcorr = xcorr[len(xcorr)//2]
                
                # Standard correlation and covariance
                corr_matrix = np.corrcoef(F1_position, F2_position)
                F1xF2_corr = corr_matrix[0, 1] if corr_matrix.shape == (2, 2) else 0
                
                cov_matrix = np.cov(F1_position, F2_position)
                F1xF2_cov = cov_matrix[0, 1] if cov_matrix.shape == (2, 2) else 0
            else:
                F1xF2_xcorr = F1xF2_corr = F1xF2_cov = 0

            # Calculate ratio
            if offset_freq[0] != 0 and offset_freq[1] != 0:
                ratio = onset_freq / offset_freq
            else:
                ratio = np.array([1, 1])
            
            # Create output row
            this_row = pd.DataFrame([[
                participant, task, full_duration,
                onset_freq[0], onset_freq[1],
                offset_freq[0], offset_freq[1], 
                range_mat[0], range_mat[1],
                slope_mat[0], slope_mat[1],
                F1xF2_xcorr, F1xF2_corr, F1xF2_cov,
                ratio[0], ratio[1], 
                F1_vel, F1_accel, F1_jerk,
                F2_vel, F2_accel, F2_jerk
            ]], columns=columns)

            results_table = pd.concat([results_table, this_row], ignore_index=True)

    print(f"\nProcessed formant summary:")
    print(f"  Total rows: {len(results_table)}")
    if len(results_table) > 0:
        print(f"  F1Accel range: {results_table['F1Accel'].min():.2f} to {results_table['F1Accel'].max():.2f}")
        print(f"  F1Slope range: {results_table['Slope_1'].min():.4f} to {results_table['Slope_1'].max():.4f}")
    
    return results_table

def extract_actual_vot():
    """Extract real VOT values from TextGrid files"""
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
    
    vot_data = []
    
    for tg_file in textgrid_files:
        try:
            tg = TextGrid.fromFile(tg_file)
            participant = os.path.basename(tg_file).replace('.TextGrid', '')
            
            vowel_tier = tg.tiers[0]
            
            for interval in vowel_tier.intervals:
                label = interval.mark.strip().lower()
                
                if 'vot' in label:
                    vot_duration = interval.maxTime - interval.minTime
                    
                    consonant = None
                    if 'p' in label:
                        consonant = 'p'
                    elif 't' in label:
                        consonant = 't'
                    elif 'k' in label:
                        consonant = 'k'
                    
                    if consonant:
                        rep_match = re.search(r'(\d+)', label)
                        rep = int(rep_match.group(1)) if rep_match else 1
                        task = f"{consonant}uh{rep}"
                        
                        vot_data.append({
                            'Participant': participant,
                            'Task': task,
                            'VOT_actual': vot_duration
                        })
                        
        except Exception as e:
            print(f"Error extracting VOT from {tg_file}: {e}")
    
    return pd.DataFrame(vot_data)

def enhance_formant_features(formant_data):
    """Add derived features with proper calculations"""
    
    # Rename columns
    rename_map = {
        "OnsetFreq_1":  "F1OnsetFreq",
        "OnsetFreq_2":  "F2OnsetFreq",
        "OffsetFreq_1": "F1OffsetFreq",
        "OffsetFreq_2": "F2OffsetFreq",
        "Range_1":      "F1Range",
        "Range_2":      "F2Range",
        "Slope_1":      "F1Slope",
        "Slope_2":      "F2Slope",
        "Ratio_1":      "F1Ratio",
        "Ratio_2":      "F2Ratio"
    }
    formant_data = formant_data.copy()
    formant_data.rename(columns=rename_map, inplace=True)
    
    # Convert to numeric
    numeric_cols = [
        "F1OnsetFreq", "F2OnsetFreq", "F1OffsetFreq", "F2OffsetFreq",
        "F1Range", "F2Range", "F1Slope", "F2Slope", "F1Ratio", "F2Ratio",
        "F1Vel", "F1Accel", "F1Jerk", "F2Vel", "F2Accel", "F2Jerk"
    ]
    
    for col in numeric_cols:
        if col in formant_data.columns:
            formant_data[col] = pd.to_numeric(formant_data[col], errors="coerce")
    
    # Extract actual VOT values
    vot_df = extract_actual_vot()
    
    if not vot_df.empty:
        formant_data = pd.merge(
            formant_data, 
            vot_df, 
            on=['Participant', 'Task'], 
            how='left'
        )
        formant_data["VOT"] = formant_data["VOT_actual"].fillna(0.05)
        formant_data.drop(columns=["VOT_actual"], inplace=True, errors='ignore')
    else:
        formant_data["VOT"] = 0.05
    
    # Create derived features
    formant_data["Syll"] = formant_data["VOT"] + formant_data["Vow"]
    formant_data["ConSpace"] = formant_data["F2OnsetFreq"] - formant_data["F1OnsetFreq"]
    formant_data["VowSpace"] = formant_data["F2OffsetFreq"] - formant_data["F1OffsetFreq"]
    formant_data["F1Range"] = formant_data["F1Range"].abs()
    formant_data["F2Range"] = formant_data["F2Range"].abs()
    
    # Add proportion features
    formant_data["VOTVowProp"] = formant_data["VOT"] / (formant_data["Vow"] + 1e-10)
    formant_data["VOTSyllProp"] = formant_data["VOT"] / (formant_data["Syll"] + 1e-10)
    formant_data["VowSyllProp"] = formant_data["Vow"] / (formant_data["Syll"] + 1e-10)
    
    # Extract repetition number
    formant_data['Rep'] = formant_data['Task'].str.extract(r'(\d+)').fillna(1).astype(int)
    
    # Features for variability calculation
    var_cols = [
        "VOT", "Vow", "Syll",
        "VOTVowProp", "VOTSyllProp", "VowSyllProp",
        "F1OnsetFreq", "F2OnsetFreq", "ConSpace",
        "F1OffsetFreq", "F2OffsetFreq", "VowSpace",
        "F1Range", "F2Range",
        "F1Slope", "F2Slope",
        "F1xF2Xcorr", "F1xF2Corr", "F1xF2Cov",
        "F1Ratio", "F2Ratio",
        "F1Vel", "F1Accel", "F1Jerk",
        "F2Vel", "F2Accel", "F2Jerk"
    ]
    
    # Calculate PhonVar
    if TASK_TYPE == "SMR":
        group_cols = ["Participant", "Rep"]
    else:
        formant_data['Consonant'] = formant_data['Task'].str.extract(r'([ptk])')
        group_cols = ["Participant", "Consonant"]
    
    for c in var_cols:
        if c in formant_data.columns:
            formant_data[f"PhonVar_{c}"] = formant_data.groupby(group_cols)[c].transform(
                lambda x: x.std(ddof=1) if len(x) > 1 else 0
            )
    
    # Calculate RepVar
    formant_data['TaskBase'] = formant_data['Task'].str.replace(r'\d+', '', regex=True)
    group_cols_2 = ["Participant", "TaskBase"]
    
    for c in var_cols:
        if c in formant_data.columns:
            std_col = formant_data.groupby(group_cols_2)[c].transform(
                lambda x: x.std(ddof=1) if len(x) > 1 else 0
            )
            mean_col = formant_data.groupby(group_cols_2)[c].transform('mean')
            formant_data[f"RepVar_{c}"] = np.where(
                mean_col.abs() > 1e-10,
                (std_col / mean_col.abs()) * 100,
                0
            )
    
    # Clean up
    formant_data.drop(columns=['TaskBase'], inplace=True, errors='ignore')
    if 'Consonant' in formant_data.columns:
        formant_data.drop(columns=['Consonant'], inplace=True, errors='ignore')
    
    return formant_data

# ============================================================================
# 3. DURATION EXTRACTION (keeping same as original)
# ============================================================================

import re

def get_ddk_tier(tg):
    """Find the tier that holds the DDK span (e.g., 'full', 'ddk', 'DDKrate')."""
    candidates = {"full", "ddk", "ddkrate", "ddk_rate", "rate"}
    for tier in tg.tiers:
        name = (getattr(tier, "name", "") or "").strip().lower()
        if name in candidates or any(k in name for k in ("ddk", "rate", "full")):
            return tier
    return None

def get_amr_syllable_tier(tg):
    """Prefer 'consonant' for per-rep AMR durations; fallback to 'reps', then 'vowel'."""
    preferences = ["consonant", "reps", "vowel"]
    for want in preferences:
        for tier in tg.tiers:
            name = (getattr(tier, "name", "") or "").strip().lower()
            if name == want:
                return tier
    return None


def main_duration():
    tier_index = 3  # keep as-is for SMR branch
    results = []
    
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
    
    for tg_path in textgrid_files:
        filename = os.path.basename(tg_path)
        base = os.path.splitext(filename)[0]
        
        try:
            tg = TextGrid.fromFile(tg_path)

            # ---- AMR path: robust tier detection + per-interval durations ----
            if TASK_TYPE == "AMR":
                # find DDK span
                ddk_tier = get_ddk_tier(tg)
                total_duration = None
                if ddk_tier:
                    for ddk_interval in ddk_tier:
                        label = (ddk_interval.mark or "").strip().lower()
                        norm  = re.sub(r'[^a-z0-9]+', '', label)
                        if norm.startswith("ddk"):
                            total_duration = ddk_interval.maxTime - ddk_interval.minTime
                            break

                # pick a tier to iterate syllables
                syl_tier = get_amr_syllable_tier(tg)
                if not syl_tier:
                    print(f"⚠️ No consonant/reps/vowel tier found in {filename}")
                    continue

                # collect labeled syllables and their durations
                labeled_intervals = []
                for interval in syl_tier:
                    raw = (interval.mark or "").strip().lower()
                    if not raw:
                        continue
                    consonant, rep = parse_amr_label(raw)
                    if consonant:
                        dur = interval.maxTime - interval.minTime
                        task_name = f"{consonant}uh{rep}"
                        labeled_intervals.append((task_name, dur))

                # DDKRate: reps per DDK span if available, else approximate
                total_reps = len(labeled_intervals)
                if total_duration and total_duration > 0 and total_reps > 0:
                    ddkrate = total_reps / total_duration
                else:
                    if labeled_intervals:
                        median_dur = float(np.median([d for _, d in labeled_intervals]))
                        ddkrate = (1.0 / median_dur) if median_dur > 0 else 0.0
                    else:
                        ddkrate = 0.0

                # emit rows
                for task_name, dur in labeled_intervals:
                    results.append({
                        "Participant": base,
                        "Task": task_name,
                        "Duration": dur,
                        "DDKRate": ddkrate
                    })

            # ---- SMR path: leave exactly as you had it ----
            else:
                if len(tg.tiers) > 3:
                    full_tier = tg.tiers[3]
                    for interval in full_tier:
                        if 'ddk' in interval.mark.lower():
                            total_duration = interval.maxTime - interval.minTime
                            ddkrate = 9 / total_duration
                            
                            for rep in range(1, 4):
                                for consonant, offset in [('p', 0), ('t', 1), ('k', 2)]:
                                    task_name = f"{consonant}uh{rep}"
                                    syllable_duration = total_duration / 9
                                    
                                    results.append({
                                        "Participant": base,
                                        "Task": task_name,
                                        "Duration": syllable_duration,
                                        "DDKRate": ddkrate
                                    })
                            break
        
        except Exception as e:
            print(f"Error processing duration for {filename}: {e}")
    
    duration_df = pd.DataFrame(results)
    duration_file = os.path.join(output_dir, "Duration_Data_For_Spreadsheets.csv")
    duration_df.to_csv(duration_file, index=False)
    return duration_df

# ============================================================================
# 4-7. KEEP OTHER EXTRACTORS THE SAME (Spectrum, Gap, Ratio, Merge)
# ============================================================================

def manual_pre_emphasize(sound, alpha=0.97):
    values = sound.values.flatten()
    pre_emphasized_values = np.append(values[0], values[1:] - alpha * values[:-1])
    sound.values[:] = pre_emphasized_values.reshape(sound.values.shape)
    return sound

def main_spectrum():
    ENERGY_THRESHOLD = 1e-5
    PAD_DURATION = 0.2
    results = []
    
    file_list = [f for f in os.listdir(project_dir) if f.endswith(".TextGrid")]
    if MAX_FILES:
        file_list = file_list[:MAX_FILES]
    
    for textgrid_file in file_list:
        base_filename = os.path.splitext(textgrid_file)[0]
        wav_file = os.path.join(project_dir, base_filename + ".wav")
        textgrid_path = os.path.join(project_dir, textgrid_file)
        
        try:
            sound = parselmouth.Sound(wav_file)
            tg = TextGrid.fromFile(textgrid_path)
            
            if len(tg.tiers) < 2:
                continue
            
            consonant_tier = tg.tiers[1]
            
            for interval in consonant_tier.intervals:
                phoneme = interval.mark.strip()
                if not phoneme:
                    continue
                
                if TASK_TYPE == "AMR":
                    consonant, rep = parse_amr_label(phoneme)
                    if not consonant:
                        continue
                    task_name = f"{consonant}uh{rep}"
                else:
                    phoneme_lower = phoneme.lower()
                    
                    if 'p' in phoneme_lower:
                        consonant = 'p'
                    elif 't' in phoneme_lower:
                        consonant = 't'
                    elif 'k' in phoneme_lower:
                        consonant = 'k'
                    else:
                        continue
                    
                    rep_match = re.search(r'(\d+)', phoneme_lower)
                    rep = int(rep_match.group(1)) if rep_match else 1
                    task_name = f"{consonant}uh{rep}"
                
                start_time = interval.minTime
                end_time = interval.maxTime
                
                if start_time >= end_time or (end_time - start_time) < 0.01:
                    continue
                
                extended_start_time = max(0, start_time - PAD_DURATION)
                extended_end_time = min(sound.xmax, end_time + PAD_DURATION)
                
                try:
                    sound_interval = sound.extract_part(
                        from_time=extended_start_time,
                        to_time=extended_end_time,
                        window_shape=parselmouth.WindowShape.HAMMING,
                        preserve_times=True
                    )
                    
                    if sound_interval is None or sound_interval.get_energy() < ENERGY_THRESHOLD:
                        continue
                    
                    sound_interval.scale_intensity(70)
                    sound_preemphasized = manual_pre_emphasize(sound_interval)
                    
                    if sound_preemphasized.get_total_duration() >= 0.05:
                        spectrum = sound_preemphasized.to_spectrum()
                        central_gravity = spectrum.get_centre_of_gravity(power=2)
                        std_deviation = spectrum.get_standard_deviation(power=2)
                        skewness = spectrum.get_skewness(power=2)
                        kurtosis = spectrum.get_kurtosis(power=2)
                        
                        results.append([
                            base_filename,
                            task_name,
                            rep,
                            central_gravity,
                            std_deviation,
                            skewness,
                            kurtosis
                        ])
                
                except Exception as e:
                    continue
        
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results, columns=[
        "Participant", "Task", "Rep", "CentralGravity", "StandardDeviation", "Skewness", "Kurtosis"
    ])
    
    spectrum_file = os.path.join(output_dir, "Spectrum_Data_For_Spreadsheets.csv")
    results_df.to_csv(spectrum_file, index=False)
    return results_df

def main_gap():
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
    
    gap_results = []
    
    for textgrid_file in textgrid_files:
        filename = os.path.basename(textgrid_file)
        base_name = os.path.splitext(filename)[0]
        
        try:
            tg = TextGrid.fromFile(textgrid_file)
            
            if len(tg.tiers) > 0:
                vowel_tier = tg.tiers[0]
                
                if TASK_TYPE == "AMR":
                    consonant_groups = {'p': [], 't': [], 'k': []}
                    
                    for interval in vowel_tier.intervals:
                        label = interval.mark.strip()
                        if label and 'vot' not in label.lower():
                            consonant, rep = parse_amr_label(label)
                            if consonant:
                                consonant_groups[consonant].append({
                                    'start': interval.minTime,
                                    'end': interval.maxTime,
                                    'rep': rep
                                })
                    
                    for consonant, intervals in consonant_groups.items():
                        if len(intervals) > 1:
                            intervals.sort(key=lambda x: x['rep'])
                            
                            gaps = []
                            for i in range(len(intervals) - 1):
                                gap = intervals[i + 1]['start'] - intervals[i]['end']
                                gaps.append(gap)
                            
                            mean_gap = np.mean(gaps) if gaps else 0
                            
                            for interval_data in intervals:
                                task_name = f"{consonant}uh{interval_data['rep']}"
                                gap_results.append({
                                    "Participant": base_name,
                                    "Task": task_name,
                                    "Gap": mean_gap
                                })
                
                else:  # SMR
                    syllable_intervals = []
                    
                    for interval in vowel_tier.intervals:
                        label = interval.mark.strip().lower()
                        if label and 'vot' not in label:
                            if 'puh' in label:
                                consonant = 'p'
                            elif 'tuh' in label:
                                consonant = 't'
                            elif 'kuh' in label:
                                consonant = 'k'
                            else:
                                continue
                            
                            rep_match = re.search(r'(\d+)', label)
                            rep = int(rep_match.group(1)) if rep_match else 1
                            
                            syllable_intervals.append({
                                'consonant': consonant,
                                'rep': rep,
                                'start': interval.minTime,
                                'end': interval.maxTime
                            })
                    
                    gaps = []
                    if len(syllable_intervals) > 1:
                        syllable_intervals.sort(key=lambda x: (x['rep'], ['p', 't', 'k'].index(x['consonant'])))
                        for i in range(len(syllable_intervals) - 1):
                            gap = syllable_intervals[i + 1]['start'] - syllable_intervals[i]['end']
                            gaps.append(gap)
                    
                    mean_gap = np.mean(gaps) if gaps else 0.0
                    
                    for rep in range(1, 4):
                        for consonant in ['p', 't', 'k']:
                            task_name = f"{consonant}uh{rep}"
                            gap_results.append({
                                "Participant": base_name,
                                "Task": task_name,
                                "Gap": mean_gap
                            })
        
        except Exception as e:
            print(f"Error processing gaps for {filename}: {e}")
    
    gap_df = pd.DataFrame(gap_results)
    gap_file = os.path.join(output_dir, "Gap_Data_For_Spreadsheets.csv")
    gap_df.to_csv(gap_file, index=False)
    return gap_df

def main_ratio():
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
    
    ratio_results = []
    
    for textgrid_file in textgrid_files:
        filename = os.path.basename(textgrid_file)
        base_name = os.path.splitext(filename)[0]
        
        try:
            tg = TextGrid.fromFile(textgrid_file)
            
            if len(tg.tiers) > 0:
                vowel_tier = tg.tiers[0]
                vot_times = []
                
                for interval in vowel_tier.intervals:
                    label = interval.mark.strip().lower()
                    if 'vot' in label:
                        if TASK_TYPE == "AMR":
                            consonant, rep = parse_amr_label(label)
                            if consonant:
                                vot_times.append({
                                    'time': interval.minTime,
                                    'consonant': consonant,
                                    'rep': rep
                                })
                        else:
                            if 'p' in label:
                                consonant = 'p'
                            elif 't' in label:
                                consonant = 't'
                            elif 'k' in label:
                                consonant = 'k'
                            else:
                                continue
                            
                            rep_match = re.search(r'(\d+)', label)
                            rep = int(rep_match.group(1)) if rep_match else 1
                            
                            vot_times.append({
                                'time': interval.minTime,
                                'consonant': consonant,
                                'rep': rep
                            })
                
                if vot_times:
                    if TASK_TYPE == "AMR":
                        consonant_groups = {}
                        for vot in vot_times:
                            if vot['consonant'] not in consonant_groups:
                                consonant_groups[vot['consonant']] = []
                            consonant_groups[vot['consonant']].append(vot)
                        
                        for consonant, group in consonant_groups.items():
                            if len(group) > 1:
                                group.sort(key=lambda x: x['rep'])
                                intervals = []
                                for i in range(len(group) - 1):
                                    intervals.append(group[i + 1]['time'] - group[i]['time'])
                                
                                mean_interval = np.mean(intervals) if intervals else 1.0
                                
                                for i, vot_data in enumerate(group):
                                    if i < len(intervals):
                                        burst_interval = intervals[i]
                                    else:
                                        burst_interval = mean_interval
                                    
                                    duration_ratio = burst_interval / mean_interval if mean_interval > 0 else 1.0
                                    
                                    task_name = f"{consonant}uh{vot_data['rep']}"
                                    ratio_results.append({
                                        "Participant": base_name,
                                        "Task": task_name,
                                        "BurstToBurst": burst_interval,
                                        "DurationRatio": duration_ratio,
                                        "DistanceFrom1": duration_ratio - 1.0
                                    })
                    else:
                        if len(vot_times) > 1:
                            vot_times.sort(key=lambda x: (x['rep'], ['p', 't', 'k'].index(x['consonant'])))
                            
                            all_intervals = []
                            for i in range(len(vot_times) - 1):
                                all_intervals.append(vot_times[i + 1]['time'] - vot_times[i]['time'])
                            
                            mean_interval = np.mean(all_intervals) if all_intervals else 0.2
                            
                            vot_map = {}
                            for vot in vot_times:
                                key = f"{vot['consonant']}uh{vot['rep']}"
                                vot_map[key] = vot
                            
                            interval_index = 0
                            for rep in range(1, 4):
                                for consonant in ['p', 't', 'k']:
                                    task_name = f"{consonant}uh{rep}"
                                    
                                    if task_name in vot_map and interval_index < len(all_intervals):
                                        burst_interval = all_intervals[interval_index]
                                        interval_index += 1
                                    else:
                                        burst_interval = mean_interval
                                    
                                    duration_ratio = burst_interval / mean_interval if mean_interval > 0 else 1.0
                                    
                                    ratio_results.append({
                                        "Participant": base_name,
                                        "Task": task_name,
                                        "BurstToBurst": burst_interval,
                                        "DurationRatio": duration_ratio,
                                        "DistanceFrom1": duration_ratio - 1.0
                                    })
                        else:
                            for rep in range(1, 4):
                                for consonant in ['p', 't', 'k']:
                                    task_name = f"{consonant}uh{rep}"
                                    ratio_results.append({
                                        "Participant": base_name,
                                        "Task": task_name,
                                        "BurstToBurst": 0.2,
                                        "DurationRatio": 1.0,
                                        "DistanceFrom1": 0.0
                                    })
                else:
                    for rep in range(1, 4):
                        for consonant in ['p', 't', 'k']:
                            task_name = f"{consonant}uh{rep}"
                            ratio_results.append({
                                "Participant": base_name,
                                "Task": task_name,
                                "BurstToBurst": 0.2,
                                "DurationRatio": 1.0,
                                "DistanceFrom1": 0.0
                            })
        
        except Exception as e:
            print(f"Error processing ratios for {filename}: {e}")
            for rep in range(1, 4):
                for consonant in ['p', 't', 'k']:
                    task_name = f"{consonant}uh{rep}"
                    ratio_results.append({
                        "Participant": base_name,
                        "Task": task_name,
                        "BurstToBurst": 0.2,
                        "DurationRatio": 1.0,
                        "DistanceFrom1": 0.0
                    })
    
    ratio_df = pd.DataFrame(ratio_results)
    ratio_file = os.path.join(output_dir, "Ratio_Data_For_Spreadsheets.csv")
    ratio_df.to_csv(ratio_file, index=False)
    return ratio_df

def merge_all_features():
    files_to_load = {
        'cpp': 'Cepstrum_Data_For_Spreadsheets.csv',
        'formant': 'Formant_Data_For_Spreadsheets.csv',
        'duration': 'Duration_Data_For_Spreadsheets.csv',
        'spectrum': 'Spectrum_Data_For_Spreadsheets.csv',
        'gap': 'Gap_Data_For_Spreadsheets.csv',
        'ratio': 'Ratio_Data_For_Spreadsheets.csv'
    }
    
    dataframes = {}
    for name, filename in files_to_load.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if len(df) > 0:
                    print(f"Loaded {name}: {len(df)} rows, {len(df.columns)} columns")
                    
                    if 'Rep' in df.columns and name != 'formant':
                        df = df.drop(columns=['Rep'])
                    
                    dataframes[name] = df
            except pd.errors.EmptyDataError:
                print(f"Skipping empty file: {filename}")
        else:
            print(f"File not found: {filename}")
    
    if 'formant' in dataframes:
        merged_df = dataframes['formant'].copy()
        
        for name, df in dataframes.items():
            if name == 'formant':
                continue
            
            rename_dict = {}
            for col in df.columns:
                if col not in ['Participant', 'Task']:
                    rename_dict[col] = f"{name}_{col}"
            
            df_renamed = df.rename(columns=rename_dict)
            merged_df = pd.merge(merged_df, df_renamed, on=['Participant', 'Task'], how='outer')
    else:
        print("No formant data found")
        merged_df = pd.DataFrame()
    
    merged_file = os.path.join(output_dir, f"MERGED_All_Features_{TASK_TYPE}.csv")
    merged_df.to_csv(merged_file, index=False)
    
    print(f"\nMerged file saved to {merged_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    
    return merged_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n1. Extracting CPP features...")
    cpp_data = main_cpp()
    
    print("\n2. Extracting Formant features...")
    formant_raw = main_formant()
    print(f"Raw formant frames: {len(formant_raw)}")
    
    # Only run diagnostics if data exists
    if len(formant_raw) > 0:
        print("\n=== TIME VECTOR DIAGNOSIS ===")
        first_task = formant_raw[formant_raw['Task'] == formant_raw['Task'].iloc[0]]
        time_vec = first_task['Time'].values
        
        if len(time_vec) > 1:
            dt = np.diff(time_vec)
            print(f"Your data:")
            print(f"  Time step: {np.mean(dt):.6f} seconds")
            print(f"  Sampling rate: {1/np.mean(dt):.1f} Hz")
        
        formant_processed = process_formant_data(formant_raw)
        print(f"Processed formant rows: {len(formant_processed)}")
        
        if not formant_processed.empty:
            print("\nDEBUG - Checking values:")
            print(f"F1Slope range: {formant_processed['Slope_1'].min():.6f} to {formant_processed['Slope_1'].max():.6f}")
            print(f"F1Accel range: {formant_processed['F1Accel'].min():.2f} to {formant_processed['F1Accel'].max():.2f}")
        
        formant_enhanced = enhance_formant_features(formant_processed)
        formant_file = os.path.join(output_dir, "Formant_Data_For_Spreadsheets.csv")
        formant_enhanced.to_csv(formant_file, index=False)
        print(f"Enhanced formant features: {len(formant_enhanced.columns)} columns")
        
        print("\n=== ROW COUNT DEBUG ===")
        print("Unique participants in formant data:")
        for p in formant_enhanced['Participant'].unique():
            count = len(formant_enhanced[formant_enhanced['Participant'] == p])
            print(f"  {p}: {count} rows")
    else:
        print("ERROR: No formant data extracted!")
        formant_enhanced = pd.DataFrame()
    
    print("\n3. Extracting Duration features...")
    duration_data = main_duration()
    
    print("\n4. Extracting Spectrum features...")
    spectrum_data = main_spectrum()
    
    print("\n5. Extracting Gap features...")
    gap_data = main_gap()
    
    print("\n6. Extracting Ratio features...")
    ratio_data = main_ratio()
    
    print("\n7. Merging all features...")
    merged_data = merge_all_features()
    
    print("\nProcessing complete!")
    print(f"All output files saved to: {output_dir}")




