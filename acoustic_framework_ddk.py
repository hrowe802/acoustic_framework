#!/usr/bin/env python3
"""
Complete Acoustic Analysis Pipeline for AMR/SMR Data
Updated to work with single "segment" tier TextGrids

Usage: python acoustic_framework_ddk.py [AMR/SMR] [data_dir]
Example: python acoustic_framework_ddk.py SMR data/
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# SLURM SUBMISSION CHECK
# ============================================================================

# Check if running on login node and auto-submit to SLURM
if 'SLURM_JOB_ID' not in os.environ:
    import subprocess
    import tempfile
    
    # Prepare SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=acoustic_ddk
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=acoustic_framework_ddk_%j.out
#SBATCH --error=acoustic_framework_ddk_%j.err

# Load modules and activate environment
module load python/3.9
source ~/envs/ddk/bin/activate

# Run the acoustic analysis script with the same arguments
python {os.path.abspath(__file__)} {' '.join(sys.argv[1:])}
"""
    
    # Write SLURM script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(slurm_script)
        temp_script = f.name
    
    # Submit to SLURM
    try:
        result = subprocess.run(['sbatch', temp_script], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Job submitted to SLURM: {result.stdout.strip()}")
        else:
            print(f"Error submitting job: {result.stderr}")
    finally:
        os.unlink(temp_script)
    
    sys.exit(0)

# ============================================================================
# IMPORT STATEMENTS (only executed on compute node)
# ============================================================================

import parselmouth
import pandas as pd
import numpy as np
import csv
import glob
from parselmouth.praat import call
from textgrid import TextGrid
import re
from scipy import signal
import gc

def load_gender_map(project_dir):
    """
    Look for gender_list.csv in project_dir. 
    Expect columns: pid, gender (M/F). Returns list of tuples (pid_lower, gender_upper).
    Matching is done via substring search on the filename base.
    """
    csv_path = os.path.join(project_dir, "gender_list.csv")
    pairs = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not {"pid", "gender"}.issubset(set(df.columns.str.lower())):
                warnings.warn("[gender_list.csv] Missing required columns 'pid' and 'gender'. Ignoring file.")
                return pairs
            cols = {c.lower(): c for c in df.columns}
            for _, row in df.iterrows():
                pid = str(row[cols["pid"]]).strip()
                gender = str(row[cols["gender"]]).strip().upper()
                if pid and gender in {"M", "F"}:
                    pairs.append((pid.lower(), gender))
        except Exception as e:
            warnings.warn(f"[gender_list.csv] Failed to read/parse: {e}")
    return pairs

def infer_gender_for_file(base_name, gender_pairs):
    """
    Return 'M' or 'F' if we can infer, else None.
    - Try CSV substring matches (first match wins)
    - Fallback to filename suffixes _M / _F (case-insensitive)
    """
    b = base_name.lower()
    for pid, g in gender_pairs:
        if pid and pid in b:
            return g
    if re.search(r"(?:^|[_-])m(?:$|[_-])", b):
        return "M"
    if re.search(r"(?:^|[_-])f(?:$|[_-])", b):
        return "F"
    return None

def max_formant_for_gender(gender, default_female_hz=5500, male_hz=5000):
    """Map gender -> Praat 'max formant'. Unknown => default 5500."""
    return male_hz if gender == "M" else default_female_hz

# ============================================================================
# COMMAND LINE ARGUMENT PROCESSING
# ============================================================================

def print_usage():
    print("Usage: python acoustic_framework.py [AMR/SMR] [data_dir]")
    print("  AMR/SMR: Task type (required)")
    print("  data_dir: Directory containing TextGrid/WAV files")
    print("\nExample: python acoustic_framework.py SMR data/")
    sys.exit(1)

# Parse command line arguments
if len(sys.argv) < 3:
    print("Error: Not enough arguments provided.")
    print_usage()

TASK_TYPE = sys.argv[1].upper()
if TASK_TYPE not in ['AMR', 'SMR']:
    print(f"Error: Task type must be 'AMR' or 'SMR', got '{sys.argv[1]}'")
    print_usage()

MAX_FILES = None  # Always process all files

data_dir = sys.argv[2]
project_dir = data_dir
output_dir = data_dir

# Extract project name from path (two levels up from output directory)
PROJECT = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(output_dir))))

# Always look for gender_list.csv in parent DDK directory
ddk_dir = os.path.dirname(os.path.abspath(project_dir))
GENDER_PAIRS = load_gender_map(ddk_dir)
if GENDER_PAIRS:
    print(f"Gender list loaded with {len(GENDER_PAIRS)} entries.")
else:
    print("No gender_list.csv found or no valid entries; using filename hints (_M/_F) or defaults.")

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

def parse_smr_label(label):
    """Parse SMR labels like 'p01', 't01', 'k01' to extract consonant and rep number"""
    label = label.lower().strip()
    
    # Match patterns like p01, t01, k01, puh01, tuh01, kuh01
    match = re.match(r'([ptk])(?:uh)?(\d+)', label)
    if match:
        consonant = match.group(1)
        rep_num = int(match.group(2))
        return consonant, rep_num
    
    return None, None

def get_tier_by_name(tg, preferred_names):
    """Find a tier by preferred names (case-insensitive)"""
    for want in preferred_names:
        for tier in tg.tiers:
            name = (getattr(tier, "name", "") or "").strip().lower()
            if name == want.lower():
                return tier
    return None

# ============================================================================
# 1. CEPSTRUM EXTRACTION
# ============================================================================

def calculate_cepstrum(snd, onset, offset):
    sound_part = snd.extract_part(from_time=onset, to_time=offset, 
                                  window_shape=parselmouth.WindowShape.HAMMING, 
                                  relative_width=1.988)
    parselmouth.praat.call(sound_part, "To Formant (burg)", 0, 5, 5000, 0.0025, 50)
    
    power_cepstrogram = parselmouth.praat.call(sound_part, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    parselmouth.praat.call(power_cepstrogram, "Smooth", 0.02, 0.0005)
    
    cepstrum = parselmouth.praat.call(power_cepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 
                                  60, 330, 0.05, "Parabolic", 0.001, 0, 
                                  "Exponential decay", "Robust")
    return cepstrum

def process_file_cepstrum(textgrid_path, snd, participant_name):
    """
    Cepstrum extraction using segment tier only.
    Calculates DDK span from first VOT to last syllable end.
    """
    out = []
    tg = TextGrid.fromFile(textgrid_path)

    # Get segment tier
    segment_tier = get_tier_by_name(tg, ["segment"])
    if not segment_tier and len(tg.tiers) > 0:
        segment_tier = tg.tiers[0]
    
    if not segment_tier:
        print(f"Warning: No segment tier found for {participant_name}")
        return out

    # Find DDK span from segment tier data
    vot_starts = []
    syllable_ends = []
    segments_found = []
    
    for iv in segment_tier.intervals:
        lab = (iv.mark or "").strip()
        if not lab or iv.maxTime <= iv.minTime:
            continue
        
        if 'vot' in lab.lower():
            vot_starts.append(iv.minTime)
        else:
            syllable_ends.append(iv.maxTime)
            if TASK_TYPE == "AMR":
                c, rep = parse_amr_label(lab)
            else:
                c, rep = parse_smr_label(lab)
            if c and rep:
                segments_found.append((f"{c}uh{rep:02d}", rep))

    # Calculate DDK span
    if vot_starts and syllable_ends:
        ddk_span = (min(vot_starts), max(syllable_ends))
    else:
        print(f"Warning: Could not determine DDK span for {participant_name}")
        return out

    # Compute cepstrum over the full span
    onset, offset = ddk_span
    try:
        cepstrum = calculate_cepstrum(snd, onset, offset)
    except Exception as e:
        print(f"Cepstrum error in {participant_name}: {e}")
        return out

    # Emit one row per actual segment label
    if segments_found:
        for task, rep in segments_found:
            out.append([participant_name, task, rep, cepstrum])
    else:
        # Default output if no segments found
        for rep in range(1, 4):
            for c in ['p', 't', 'k']:
                out.append([participant_name, f"{c}uh{rep:02d}", rep, cepstrum])

    return out

def main_cepstrum():
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
            output_rows.extend(process_file_cepstrum(textgrid_path, snd, participant_name))
            files_processed += 1
            print(f"Processed cepstrum for {filename} ({files_processed}/{MAX_FILES if MAX_FILES else 'all'})")
            
            # Add these two lines to free memory:
            del snd
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    cepstral_df = pd.DataFrame(output_rows[1:], columns=output_rows[0])
    print(f"Cepstrum processing complete. {len(cepstral_df)} rows extracted.")
    return cepstral_df

# ============================================================================
# 2. FORMANT EXTRACTION
# ============================================================================

def main_formant():
    """Extract raw formant data from segment tier segments (excluding VOT)"""
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    print(f"Found {len(textgrid_files)} TextGrid files")
    
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]
        print(f"Limited to {MAX_FILES} files")
    
    results = []

    for textgrid_file in textgrid_files:
        filename = os.path.basename(textgrid_file).replace('.TextGrid', '')
        sound_path = os.path.join(project_dir, filename + ".wav")

        if not os.path.exists(sound_path):
            print(f"Sound file for {filename} not found. Skipping.")
            continue

        try:
            tg = TextGrid.fromFile(textgrid_file)
            sound = parselmouth.Sound(sound_path)
            
            # Use segment tier for formant extraction
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier:
                segment_tier = tg.tiers[0] if len(tg.tiers) > 0 else None
            
            if not segment_tier:
                print(f"No segment tier found for {filename}")
                continue
            
            # Decide max formant from gender
            gender = infer_gender_for_file(filename, GENDER_PAIRS)
            max_formant_hz = max_formant_for_gender(gender, default_female_hz=5500, male_hz=5000)

            # Set number of formants based on gender
            if gender == "M":
                num_formants = 4
            else:
                num_formants = 5  # Female or unknown

            formant = call(sound, "To Formant (burg)...",
                0,               # time step (Praat auto)
                num_formants,    # <-- gender-specific (4 for M, 5 for F)
                max_formant_hz,  # <-- gender-aware max frequency (5000 for M, 5500 for F)
                0.025,           # window length
                50)              # pre-emphasis
            
            if gender in {"M","F"}:
                print(f"[{filename}] gender={gender} -> max_formant={max_formant_hz} Hz")
            else:
                print(f"[{filename}] gender=UNKNOWN -> default max_formant={max_formant_hz} Hz")

            for interval in segment_tier.intervals:
                phoneme = interval.mark.strip()

                # Skip VOT intervals and empty labels
                if not phoneme or 'vot' in phoneme.lower():
                    continue
                
                start_time = interval.minTime
                end_time = interval.maxTime
                duration = end_time - start_time
                
                # Parse task name based on task type
                if TASK_TYPE == "AMR":
                    consonant, rep = parse_amr_label(phoneme)
                    if not consonant:
                        continue
                    task_name = f"{consonant}uh{rep:02d}"
                else:  # SMR
                    consonant, rep = parse_smr_label(phoneme)
                    if not consonant:
                        continue
                    task_name = f"{consonant}uh{rep:02d}"

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
    
    print(f"Formant processing complete. {len(df_results)} frames extracted.")
    
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
    
    return results_table

def extract_actual_vot():
    """
    Extract real VOT values from segment tier.
    VOT intervals are those with 'vot' in the label.
    """
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]

    out = []

    for tg_file in textgrid_files:
        try:
            tg = TextGrid.fromFile(tg_file)
            participant = os.path.basename(tg_file).replace('.TextGrid', '')

            # Use segment tier
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier and len(tg.tiers) > 0:
                segment_tier = tg.tiers[0]
            
            if not segment_tier:
                continue

            # Process VOT intervals
            for interval in segment_tier.intervals:
                lab = (interval.mark or "").strip()
                
                # Check if this is a VOT interval
                if 'vot' in lab.lower() and interval.maxTime > interval.minTime:
                    # Parse the VOT label to get consonant and rep
                    vot_match = re.match(r'([ptk])(\d+)vot', lab.lower())
                    if vot_match:
                        consonant = vot_match.group(1)
                        rep = int(vot_match.group(2))
                        task = f"{consonant}uh{rep:02d}"
                        vot_duration = interval.maxTime - interval.minTime
                        
                        out.append({
                            'Participant': participant,
                            'Task': task,
                            'VOT_actual': vot_duration
                        })

        except Exception as e:
            print(f"Error extracting VOT from {tg_file}: {e}")

    if out:
        df = pd.DataFrame(out)
        return df

    return pd.DataFrame(columns=['Participant', 'Task', 'VOT_actual'])

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
        missing = formant_data.merge(vot_df, on=['Participant','Task'], how='left', indicator=True)
        missing = missing[missing['_merge'] == 'left_only']
        if len(missing) > 0:
            miss_n = missing[['Participant','Task']].drop_duplicates().shape[0]
            print(f"[VOT] WARNING: {miss_n} Participant/Task pairs in formant data had no VOT match.")
    
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
# 3. DURATION EXTRACTION
# ============================================================================

def main_duration():
    """Extract per-syllable Duration and DDKRate using only segment tier.
    DDKRate calculated from first VOT start to last syllable end.
    """
    results = []

    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]

    for tg_path in textgrid_files:
        filename = os.path.basename(tg_path)
        base = os.path.splitext(filename)[0]

        try:
            tg = TextGrid.fromFile(tg_path)

            # Get segment tier
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier and len(tg.tiers) > 0:
                segment_tier = tg.tiers[0]
            
            if not segment_tier:
                print(f"WARNING: No segment tier in {filename}; no Duration rows emitted.")
                continue

            # Collect all VOT starts and syllable segments
            vot_starts = []
            syllable_segments = []
            
            for iv in segment_tier.intervals:
                raw = (iv.mark or "").strip()
                
                # Skip empty intervals
                if not raw or iv.maxTime <= iv.minTime:
                    continue
                
                # Check if it's a VOT interval
                if "vot" in raw.lower():
                    vot_starts.append(iv.minTime)
                else:
                    # It's a syllable segment
                    if TASK_TYPE == "AMR":
                        c, rep = parse_amr_label(raw)
                    else:  # SMR
                        c, rep = parse_smr_label(raw)
                    
                    if c and rep:
                        task_name = f"{c}uh{rep:02d}"
                        syllable_segments.append({
                            'task': task_name,
                            'start': iv.minTime,
                            'end': iv.maxTime,
                            'duration': iv.maxTime - iv.minTime
                        })
            
            # Calculate DDKRate from first VOT to last syllable
            ddkrate = 0.0
            if syllable_segments and vot_starts:
                # Get task boundaries
                task_start = min(vot_starts)  # First VOT start
                task_end = max(seg['end'] for seg in syllable_segments)  # Last syllable end
                total_duration = task_end - task_start
                
                # Count of syllables
                segment_count = len(syllable_segments)
                
                # Calculate rate
                if total_duration > 0:
                    ddkrate = segment_count / total_duration
                    print(f"[{base}] {segment_count} syllables in {total_duration:.3f}s = {ddkrate:.2f} syll/s")
                else:
                    print(f"WARNING: Invalid duration calculation for {filename}")
            else:
                if not vot_starts:
                    print(f"WARNING: No VOT intervals found in {filename}")
                if not syllable_segments:
                    print(f"WARNING: No valid syllables found in {filename}")
            
            # Emit duration rows for each syllable
            for seg in syllable_segments:
                results.append({
                    "Participant": base,
                    "Task": seg['task'],
                    "Duration": seg['duration'],
                    "DDKRate": ddkrate
                })

        except Exception as e:
            print(f"Error processing duration for {filename}: {e}")

    duration_df = pd.DataFrame(results)
    print(f"Duration processing complete. {len(duration_df)} rows extracted.")
    return duration_df

# ============================================================================
# 4. SPECTRUM EXTRACTION
# ============================================================================

def manual_pre_emphasize(sound, alpha=0.97):
    values = sound.values.flatten()
    pre_emphasized_values = np.append(values[0], values[1:] - alpha * values[:-1])
    sound.values[:] = pre_emphasized_values.reshape(sound.values.shape)
    return sound

def main_spectrum():
    """Extract spectrum features from first 20ms of VOT segments in segment tier"""
    ENERGY_THRESHOLD = 1e-5
    PAD_DURATION = 0.2
    VOT_ANALYSIS_DURATION = 0.020  # 20ms for consonant burst analysis
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
            
            # Use segment tier
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier and len(tg.tiers) > 0:
                segment_tier = tg.tiers[0]
            
            if not segment_tier:
                print(f"No segment tier found for {base_filename}")
                continue
            
            for interval in segment_tier.intervals:
                phoneme = interval.mark.strip()
                
                # Only process VOT intervals
                if not phoneme or 'vot' not in phoneme.lower():
                    continue
                
                # Parse the VOT label
                vot_match = re.match(r'([ptk])(\d+)vot', phoneme.lower())
                if not vot_match:
                    continue
                
                consonant = vot_match.group(1)
                rep = int(vot_match.group(2))
                task_name = f"{consonant}uh{rep:02d}"
                
                # Use first 20ms of VOT interval
                start_time = interval.minTime
                vot_duration = interval.maxTime - interval.minTime
                
                analysis_duration = min(VOT_ANALYSIS_DURATION, vot_duration)
                end_time = start_time + analysis_duration
                
                if analysis_duration < 0.005:  # Skip if too short
                    continue
                
                # Extract with padding
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
                    
                    # Extract the actual analysis window
                    analysis_sound = sound_preemphasized.extract_part(
                        from_time=start_time,
                        to_time=end_time,
                        window_shape=parselmouth.WindowShape.HAMMING,
                        preserve_times=False
                    )
                    
                    if analysis_sound.get_total_duration() >= 0.005:
                        spectrum = analysis_sound.to_spectrum()
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
                    print(f"Error processing spectrum for {task_name} in {base_filename}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing {base_filename}: {e}")
            continue
    
    spectrum_df = pd.DataFrame(results, columns=[
        "Participant", "Task", "Rep", "CentralGravity", "StandardDeviation", "Skewness", "Kurtosis"
    ])
    
    print(f"Spectrum processing complete. {len(spectrum_df)} rows extracted.")
    return spectrum_df

# ============================================================================
# 5. GAP EXTRACTION
# ============================================================================

def main_gap():
    """
    GAP extraction from segment tier only.
    GAP = time from current syllable END to next VOT START.
    """
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]

    gap_rows = []

    for tg_path in textgrid_files:
        fn = os.path.basename(tg_path)
        base = os.path.splitext(fn)[0]
        try:
            tg = TextGrid.fromFile(tg_path)
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier:
                segment_tier = tg.tiers[0] if len(tg.tiers) > 0 else None
            if not segment_tier:
                continue

            # VOT starts and Syllables
            vot_starts = []
            syll = []

            for iv in segment_tier.intervals:
                lab = (iv.mark or "").strip()
                if not lab or iv.maxTime <= iv.minTime:
                    continue
                
                if 'vot' in lab.lower():
                    vot_starts.append(iv.minTime)
                else:
                    if TASK_TYPE == "AMR":
                        c, rep = parse_amr_label(lab)
                    else:
                        c, rep = parse_smr_label(lab)
                    if c and rep:
                        syll.append((iv.minTime, iv.maxTime, f"{c}uh{rep:02d}"))

            vot_starts.sort()
            syll.sort(key=lambda x: x[0])

            if not syll:
                continue

            for (s0, s1, task) in syll:
                # Next VOT following this syllable end
                next_vot = next((v for v in vot_starts if v >= s1), None)
                gap = (next_vot - s1) if next_vot is not None else 0.0
                gap_rows.append({"Participant": base, "Task": task, "Gap": max(gap, 0.0)})

        except Exception as e:
            print(f"Error processing gaps for {fn}: {e}")

    gap_df = pd.DataFrame(gap_rows)
    print(f"Gap processing complete. {len(gap_df)} rows extracted.")
    return gap_df

# ============================================================================
# 6. RATIO EXTRACTION
# ============================================================================

def main_ratio():
    """
    Ratio extraction from segment tier only.
    Computes burst-to-burst intervals from VOT timings.
    """
    textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))
    if MAX_FILES:
        textgrid_files = textgrid_files[:MAX_FILES]

    rows = []

    for tg_path in textgrid_files:
        fn = os.path.basename(tg_path)
        base = os.path.splitext(fn)[0]
        try:
            tg = TextGrid.fromFile(tg_path)
            segment_tier = get_tier_by_name(tg, ["segment"])
            if not segment_tier:
                segment_tier = tg.tiers[0] if len(tg.tiers) > 0 else None
            if not segment_tier:
                continue

            vot_data = []

            for iv in segment_tier.intervals:
                lab = (iv.mark or "").strip()
                if not lab or iv.maxTime <= iv.minTime:
                    continue
                
                if 'vot' in lab.lower():
                    # Parse VOT label
                    vot_match = re.match(r'([ptk])(\d+)vot', lab.lower())
                    if vot_match:
                        consonant = vot_match.group(1)
                        rep = int(vot_match.group(2))
                        task = f"{consonant}uh{rep:02d}"
                        vot_data.append((iv.minTime, task, consonant, rep))

            if not vot_data:
                continue

            if TASK_TYPE == "AMR":
                # Group by consonant
                from collections import defaultdict
                by_c = defaultdict(list)
                for vt, task, c, rep in vot_data:
                    by_c[c].append((rep, vt, task))
                
                for c, items in by_c.items():
                    items.sort(key=lambda x: x[0])  # by rep
                    intervals = [items[i+1][1] - items[i][1] for i in range(len(items)-1)]
                    mean_iv = np.mean(intervals) if intervals else 1.0
                    
                    for i, (_, vt, task) in enumerate(items):
                        burst = intervals[i] if i < len(intervals) else mean_iv
                        ratio = burst/mean_iv if mean_iv > 0 else 1.0
                        rows.append({
                            "Participant": base,
                            "Task": task,
                            "BurstToBurst": burst,
                            "DurationRatio": ratio,
                            "DistanceFrom1": ratio - 1.0
                        })
            else:
                # SMR: global sequence sorted by (rep, p->t->k)
                vot_data.sort(key=lambda x: (x[3], {'p':0,'t':1,'k':2}.get(x[2], 3), x[0]))
                intervals = [vot_data[i+1][0] - vot_data[i][0] for i in range(len(vot_data)-1)]
                mean_iv = np.mean(intervals) if intervals else 0.2
                
                for i, (vt, task, c, rep) in enumerate(vot_data):
                    burst = intervals[i] if i < len(intervals) else mean_iv
                    ratio = burst/mean_iv if mean_iv > 0 else 1.0
                    rows.append({
                        "Participant": base,
                        "Task": task,
                        "BurstToBurst": burst,
                        "DurationRatio": ratio,
                        "DistanceFrom1": ratio - 1.0
                    })

        except Exception as e:
            print(f"Error processing ratios for {fn}: {e}")

    ratio_df = pd.DataFrame(rows)
    print(f"Ratio processing complete. {len(ratio_df)} rows extracted.")
    return ratio_df

# ============================================================================
# 7. MERGE ALL FEATURES
# ============================================================================

def merge_all_features(cepstrum_data, formant_data, duration_data, spectrum_data, gap_data, ratio_data):
    """Merge all feature dataframes"""
    
    dataframes = {
        'cepstrum': cepstrum_data,
        'formant': formant_data,
        'duration': duration_data,
        'spectrum': spectrum_data,
        'gap': gap_data,
        'ratio': ratio_data
    }
    
    # Filter out empty dataframes
    dataframes = {k: v for k, v in dataframes.items() if len(v) > 0}
    
    if 'formant' in dataframes:
        merged_df = dataframes['formant'].copy()
        
        for name, df in dataframes.items():
            if name == 'formant':
                continue
            
            # Drop Rep column if exists (except formant)
            if 'Rep' in df.columns:
                df = df.drop(columns=['Rep'])
            
            rename_dict = {}
            for col in df.columns:
                if col not in ['Participant', 'Task']:
                    rename_dict[col] = col
            
            df_renamed = df.rename(columns=rename_dict)
            merged_df = pd.merge(merged_df, df_renamed, on=['Participant', 'Task'], how='outer')
    else:
        print("No formant data found")
        merged_df = pd.DataFrame()
    
    merged_file = os.path.join(output_dir, f"{PROJECT}_Acoustic_Features_{TASK_TYPE}.csv")
    merged_df.to_csv(merged_file, index=False)
    
    return merged_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n1. Extracting cepstrum features...")
    cepstrum_data = main_cepstrum()
    
    print("\n2. Extracting Formant features...")
    formant_raw = main_formant()
    
    if len(formant_raw) > 0:
        formant_processed = process_formant_data(formant_raw)
        
        formant_enhanced = enhance_formant_features(formant_processed)
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
    merged_data = merge_all_features(
        cepstrum_data, 
        formant_enhanced, 
        duration_data, 
        spectrum_data, 
        gap_data, 
        ratio_data
    )
    
    print("\nProcessing complete!")
    print(f"Output file: {os.path.join(output_dir, f'{PROJECT}_Acoustic_Features_{TASK_TYPE}.csv')}")
