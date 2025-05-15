#!/usr/bin/env python
# coding: utf-8

# In[51]:


get_ipython().system('pip install praatio')


# In[52]:


import os
import parselmouth
import pandas as pd
import numpy as np
import csv
import textgrid
import glob
from parselmouth.praat import call
import os
import csv
import numpy as np
import pandas as pd
from textgrid import TextGrid
import parselmouth
import librosa
import matplotlib.pyplot as plt
import re
from praatio import textgrid


# In[3]:


project_dir = "Downloads/acoustic_framework-main-2/"
output_dir = "Downloads/TRIAL_ACOUSTIC/"  # Output directory for saving files


# # ---------------- CPP Extraction Function ---------------- #

# In[4]:


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

def process_file(textgrid_path, snd, participant_name):
    output = []
    tg = parselmouth.praat.call("Read from file", textgrid_path)
    num_intervals = parselmouth.praat.call(tg, "Get number of intervals", 3)

    for i in range(1, num_intervals + 1):
        label = parselmouth.praat.call(tg, "Get label of interval", 3, i)
        if label.startswith("puhtuhkuh"):
            onset = parselmouth.praat.call(tg, "Get start point", 3, i)
            offset = parselmouth.praat.call(tg, "Get end point", 3, i)
            cpps = calculate_cpp(snd, onset, offset)

            task, rep = label[:9], label[9] if len(label) > 9 else "N/A"
            output.append([participant_name, task, rep, cpps])
    
    return output

def main():
    output_rows = [["Participant", "Task", "Rep", "CPPS"]]
    
    for filename in os.listdir(project_dir):
        if not filename.endswith(".TextGrid"):
            continue

        textgrid_path = os.path.join(project_dir, filename)
        wav_path = textgrid_path.replace(".TextGrid", ".wav")
        participant_name = filename.replace(".TextGrid", "")

        if not os.path.exists(wav_path):
            print(f"Missing wav file for {filename}")
            continue

        try:
            snd = parselmouth.Sound(wav_path)
            output_rows.extend(process_file(textgrid_path, snd, participant_name))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save the initial CSV to the output directory
    intermediate_csv = os.path.join(output_dir, "Cepstrum_Data_For_R.csv")
    with open(intermediate_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(output_rows)

    # Expand rows and save the final output to the output directory
    cepstrum_data = pd.read_csv(intermediate_csv)
    cepstrum_data_expanded = pd.DataFrame(np.repeat(cepstrum_data.values, 3, axis=0), 
                                          columns=cepstrum_data.columns)
    
    final_csv = os.path.join(output_dir, "Cepstrum_Data_For_Spreadsheets_3.csv")
    cepstrum_data_expanded.astype(str).to_csv(final_csv, index=False)

    print("Processing complete. Final data saved to", final_csv)

if __name__ == "__main__":
    main()


# In[5]:


# ---------------- FORMANT Extraction Function PRAAT ---------------- #

# Make a list of all TextGrid files in the folder
textgrid_files = glob.glob(os.path.join(project_dir, '*.TextGrid'))

# Collect results in a list
results = []

# Loop through the list of TextGrid files
for textgrid_file in textgrid_files:
    filename = os.path.basename(textgrid_file).replace('.TextGrid', '')

    # Read the TextGrid and corresponding .wav file
    textgrid_path = os.path.join(project_dir, filename + ".TextGrid")
    sound_path = os.path.join(project_dir, filename + ".wav")

    if not os.path.exists(sound_path):
        print(f"Sound file for {filename} not found. Skipping.")
        continue

    try:
        tg = parselmouth.Data.read(textgrid_path)  # Read TextGrid
        sound = parselmouth.Sound(sound_path)      # Read corresponding sound file
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # Select the first tier (vowel tier assumption)
    try:
        number_of_intervals = call(tg, "Get number of intervals", 1)
    except Exception as e:
        print(f"Failed to get intervals for {filename}: {e}")
        continue

    # Create the Formant object
    formant = call(sound, "To Formant (burg)...", 0, 5, 5000, 0.0025, 50)

    # Loop through each interval in the first tier
    for interval_index in range(1, number_of_intervals + 1):
        phoneme = call(tg, "Get label of interval", 1, interval_index).strip()

        if phoneme != "":
            start_time = call(tg, "Get start point", 1, interval_index)
            end_time = call(tg, "Get end point", 1, interval_index)
            duration = end_time - start_time

            # Determine number of frames based on duration and frame interval (0.0025s)
            frame_count = int(duration / 0.0025)

            # Loop through each frame in the interval
            for frame in range(frame_count):
                frame_time = start_time + (frame * 0.0025)

                # Get formant values at the specified time
                f1 = call(formant, "Get value at time", 1, frame_time, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, frame_time, "Hertz", "Linear")

                # Store results in the list
                results.append([filename, phoneme, frame_time, f1, f2, duration])

# Print results as a DataFrame
df_results = pd.DataFrame(
    results,
    columns=["Participant", "Task", "Time", "F1", "F2", "Duration"]
)
print(df_results)


# # ---------------- Formant Extraction Function ---------------- #

# In[6]:


# ---------------- Formant Extraction Function MATLAB ---------------- #

def process_formant_data(df_results):
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
            task_numbers = task_data.iloc[:, 2:6].to_numpy()

            if 'VOT' in task:
                duration = task_numbers[0, 3]
                this_row = pd.DataFrame([[
                    participant, task, duration, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]], columns=columns)
                results_table = pd.concat([results_table, this_row], ignore_index=True)
                continue

            # Extract position and time for F1 and F2
            F1_position, F2_position = task_numbers[:, 1], task_numbers[:, 2]
            F1_time, F2_time = task_numbers[:, 0], task_numbers[:, 0]

            # Calculate velocity, acceleration, and jerk for F1
            F1_vel = np.diff(F1_position) / np.diff(F1_time)
            F1_accel = np.diff(F1_vel) / np.diff(F1_time[:-1])
            F1_jerk = np.diff(F1_accel) / np.diff(F1_time[:-2])

            # Apply padding after calculations
            F1_vel = np.pad(F1_vel, (0, 1), constant_values=np.nan)
            F1_accel = np.pad(F1_accel, (0, 2), constant_values=np.nan)
            F1_jerk = np.pad(F1_jerk, (0, 3), constant_values=np.nan)

            # Calculate velocity, acceleration, and jerk for F2
            F2_vel = np.diff(F2_position) / np.diff(F2_time)
            F2_accel = np.diff(F2_vel) / np.diff(F2_time[:-1])
            F2_jerk = np.diff(F2_accel) / np.diff(F2_time[:-2])

            F2_vel = np.pad(F2_vel, (0, 1), constant_values=np.nan)
            F2_accel = np.pad(F2_accel, (0, 2), constant_values=np.nan)
            F2_jerk = np.pad(F2_jerk, (0, 3), constant_values=np.nan)

            # Calculate slopes and ranges
            mid_point = len(task_numbers) // 2
            time_mat = task_numbers[:mid_point, 0]
            data_half_mat = task_numbers[:mid_point, 1:3]

            dur = time_mat[-1] - time_mat[0]
            onset_freq = data_half_mat[0, :]
            offset_freq = data_half_mat[-1, :]
            range_mat = offset_freq - onset_freq
            slope_mat = (offset_freq - onset_freq) / dur

            # Calculate correlation and covariance
            F1xF2_corr = np.corrcoef(task_numbers[:, 1], task_numbers[:, 2])[0, 1]
            F1xF2_cov = np.cov(task_numbers[:, 1], task_numbers[:, 2])[0, 1]
            F1xF2_xcorr = np.correlate(task_numbers[:, 1], task_numbers[:, 2], mode='full') / len(task_numbers[:, 1])
            F1xF2_xcorr = F1xF2_xcorr[len(task_numbers[:, 1]) - 1]

            ratio = onset_freq / offset_freq

            # Store results
            this_row = pd.DataFrame([[
                participant, task, dur, onset_freq[0], onset_freq[1],
                offset_freq[0], offset_freq[1], range_mat[0], range_mat[1],
                slope_mat[0], slope_mat[1], F1xF2_xcorr, F1xF2_corr,
                F1xF2_cov, ratio[0], ratio[1], np.nanmean(F1_vel),
                np.nanmean(F1_accel), np.nanmean(F1_jerk),
                np.nanmean(F2_vel), np.nanmean(F2_accel), np.nanmean(F2_jerk)
            ]], columns=columns)

            results_table = pd.concat([results_table, this_row], ignore_index=True)

    return results_table


# In[7]:


processed_results = process_formant_data(df_results)
print(processed_results)


# In[30]:


# ---------------- Formant Extraction Function Python ---------------- #

def vot_name_to_formant_name(vot_name):
    """Convert VOT task name to corresponding formant task name."""
    return vot_name[0] + "uh" + vot_name[1]

# Process the results_table DataFrame
def process_vot_and_formant_data(results_table):
    # Separate VOT and formant rows
    vot_rows = processed_results[processed_results['Task'].str.contains("VOT")]
    formant_rows = processed_results[~processed_results['Task'].str.contains("VOT")]

    print(f"Total data rows: {len(processed_results)}")
    print(f"Number of formant rows: {len(formant_rows)}")
    print(f"Number of VOT rows: {len(vot_rows)}")

    updated_rows = []

    for _, formant_row in formant_rows.iterrows():
        matching_vot_row = vot_rows[
            (vot_rows['Participant'] == formant_row['Participant']) & 
            (vot_rows['Task'].apply(vot_name_to_formant_name) == formant_row['Task'])
        ]

        # Assign placeholder VOT value if no match is found
        if not matching_vot_row.empty:
            formant_row['VOT'] = 1  # Placeholder value for VOT
        else:
            formant_row['VOT'] = np.nan
        
        # Extract repetition number from task name (e.g., puh2 -> 2)
        formant_row['Rep'] = formant_row['Task'][3] if len(formant_row['Task']) > 3 else "1"
        formant_row['Task'] = formant_row['Task'][:3]  # Reduce to puh, tuh, kuh
        updated_rows.append(formant_row)

    updated_results = pd.DataFrame(updated_rows)
    updated_results = updated_results[['Participant', 'Rep', 'VOT'] + 
                                      [col for col in updated_results.columns if col not in ['Participant', 'Rep', 'VOT']]]

    print(updated_results.head())
    return updated_results

# Process and save the updated DataFrame
updated_results_df = process_vot_and_formant_data(processed_results)


# In[31]:


import pandas as pd
import numpy as np
from itertools import product

# Assuming updated_results_df is the input DataFrame
formantData = updated_results_df

formantData['Task'] = formantData['Task'].str.replace('_', '', regex=False)  # Remove underscores

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
formantData.rename(columns=rename_map, inplace=True)


# 3) Convert columns to numeric
numeric_cols = [
    "F1OnsetFreq", "F2OnsetFreq", "F1OffsetFreq", "F2OffsetFreq",
    "F1Range", "F2Range", "F1Slope", "F2Slope", "F1Ratio", "F2Ratio"
]
formantData[numeric_cols] = formantData[numeric_cols].apply(pd.to_numeric, errors="coerce")

# 4) Create 'syllable' with derived metrics
formantData["Syll"] = formantData["VOT"] + formantData["Vow"]
formantData["ConSpace"] = formantData["F2OnsetFreq"] - formantData["F1OnsetFreq"]
formantData["VowSpace"] = formantData["F2OffsetFreq"] - formantData["F1OffsetFreq"]
formantData["F1Range"] = formantData["F1Range"].abs()
formantData["F2Range"] = formantData["F2Range"].abs()

syllable_cols = [
    "Participant", "Rep", "Task",
    "VOT", "Vow", "Syll",
    "F1OnsetFreq", "F2OnsetFreq", "ConSpace",
    "F1OffsetFreq", "F2OffsetFreq", "VowSpace",
    "F1Range", "F2Range",
    "F1Slope", "F2Slope",
    "F1xF2Xcorr", "F1xF2Corr", "F1xF2Cov",
    "F1Ratio", "F2Ratio",
    "F1Vel", "F1Accel", "F1Jerk",
    "F2Vel", "F2Accel", "F2Jerk"
]
syllable = formantData[syllable_cols].copy()

# 5) Calculate proportions
syllable["VOTVowProp"] = syllable["VOT"] / syllable["Vow"]
syllable["VOTSyllProp"] = syllable["VOT"] / syllable["Syll"]
syllable["VowSyllProp"] = syllable["Vow"] / syllable["Syll"]

proportions_cols = syllable_cols + ["VOTVowProp", "VOTSyllProp", "VowSyllProp"]
proportions = syllable[proportions_cols].copy()

# 6) Calculate precision
precision = proportions.copy()
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
group_cols = ["Participant", "Rep"]

for c in var_cols:
    precision[f"PhonVar_{c}"] = precision.groupby(group_cols)[c].transform("std")

# 7) Calculate precision consistency
precision_consistency = precision.copy()
group_cols_2 = ["Participant", "Task"]

for c in var_cols:
    std_col = precision_consistency.groupby(group_cols_2)[c].transform("std")
    mean_col = precision_consistency.groupby(group_cols_2)[c].transform("mean")
    precision_consistency[f"RepVar_{c}"] = (std_col / mean_col) * 100

# 8) Create 'data' for export
final_cols = [
    "Task", "VOT", "Vow", "Syll",
    "VOTVowProp", "VOTSyllProp", "VowSyllProp",
    "F1OnsetFreq", "F2OnsetFreq", "ConSpace",
    "F1OffsetFreq", "F2OffsetFreq", "VowSpace",
    "F1Range", "F2Range",
    "F1Slope", "F2Slope",
    "F1xF2Xcorr", "F1xF2Corr", "F1xF2Cov",
    "F1Ratio", "F2Ratio",
    "F1Vel", "F1Accel", "F1Jerk",
    "F2Vel", "F2Accel", "F2Jerk",
]

phonvar_cols = [f"PhonVar_{c}" for c in var_cols]
repvar_cols = [f"RepVar_{c}" for c in var_cols]
final_cols.extend(phonvar_cols)
final_cols.extend(repvar_cols)

data = (
    precision_consistency
    .groupby(["Participant", "Rep"], as_index=False)
    [final_cols]
    .apply(lambda g: g)
    .reset_index(drop=True)
)

# 9) Write to CSV
#out_path = "/Users/DELL/Downloads/acoustic_framework-main-2/Formant_Data_For_Spreadsheets.csv"
#data.to_csv(out_path, index=False)

print(data)


# In[33]:


# Append repetition numbers directly to the Task column
data['Task'] = data['Task'] + (data.groupby('Task').cumcount() + 1).astype(str)

# Verify the updated DataFrame
print(data.head())


# In[34]:


data.head()


# In[35]:


output_file = output_dir + "FORMANT_Data_For_Spreadsheets.csv"
data.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


# # ---------------- Spectrum Extraction Function ---------------- #

# In[36]:


# ---------------- SPECTRUM Extraction Function ---------------- #

# Energy threshold and padding
ENERGY_THRESHOLD = 1e-5
PAD_DURATION = 0.2  # Increased padding to 200ms

# Manual pre-emphasis filter
def manual_pre_emphasize(sound, alpha=0.97):
    values = sound.values.flatten()
    pre_emphasized_values = np.append(values[0], values[1:] - alpha * values[:-1])
    sound.values[:] = pre_emphasized_values.reshape(sound.values.shape)
    return sound

# Prepare output DataFrame
results = []

# List all TextGrid files in the directory
file_list = [f for f in os.listdir(project_dir) if f.endswith(".TextGrid")]

for textgrid_file in file_list:
    base_filename = os.path.splitext(textgrid_file)[0]
    wav_file = os.path.join(project_dir, base_filename + ".wav")
    textgrid_path = os.path.join(project_dir, textgrid_file)

    try:
        sound = parselmouth.Sound(wav_file)
        print(f"Loaded sound: {base_filename}, Duration={sound.get_total_duration()}")
        tg = TextGrid()
        tg.read(textgrid_path)
    except Exception as e:
        print(f"Error loading sound or TextGrid for {textgrid_file}: {e}")
        continue

    if len(tg.tiers) < 2:
        print(f"{textgrid_file} does not have enough tiers.")
        continue

    consonant_tier = tg.tiers[1]

    for interval in consonant_tier.intervals:
        phoneme = interval.mark.strip()
        if not phoneme:
            continue

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
                    base_filename,  # Participant
                    phoneme[0] + "uh",  # Convert short name to long name
                    phoneme[1],  # Rep
                    central_gravity,
                    std_deviation,
                    skewness,
                    kurtosis
                ])

        except Exception as e:
            print(f"Error processing {phoneme} in {textgrid_file}: {e}")
            continue

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=[
    "Participant", "Task", "Rep", "CentralGravity", "StandardDeviation", "Skewness", "Kurtosis"
])

# Reorder and save results
output_rows = []
for task in range(int(len(results_df) / 9)):
    output_rows.append(results_df.iloc[task * 9 + 2])
    output_rows.append(results_df.iloc[task * 9 + 5])
    output_rows.append(results_df.iloc[task * 9 + 8])
    output_rows.append(results_df.iloc[task * 9 + 0])
    output_rows.append(results_df.iloc[task * 9 + 3])
    output_rows.append(results_df.iloc[task * 9 + 6])
    output_rows.append(results_df.iloc[task * 9 + 1])
    output_rows.append(results_df.iloc[task * 9 + 4])
    output_rows.append(results_df.iloc[task * 9 + 7])

output_df = pd.DataFrame(output_rows)
#output_file = os.path.join(project_dir, "Spectrum_Data_For_R.csv")
#output_df.to_csv(output_file, index=False)

#print(f"Reordered results saved to {output_file}")
output_df.head()


# In[38]:


# ---------------- SPECTRUM Extraction Function R---------------- #

# Read the original data
spectrumData = output_df

# Ensure numeric columns are numeric
numeric_cols = ["CentralGravity", "StandardDeviation", "Skewness", "Kurtosis"]
spectrumData[numeric_cols] = spectrumData[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Compute phoneme-level variation (PhonVar_*) per (Participant, Rep)
grouped_pr = spectrumData.groupby(["Participant", "Rep"])
spectrumData["PhonVar_CentralGravity"] = grouped_pr["CentralGravity"].transform("std")
spectrumData["PhonVar_StandardDeviation"] = grouped_pr["StandardDeviation"].transform("std")
spectrumData["PhonVar_Skewness"] = grouped_pr["Skewness"].transform("std")
spectrumData["PhonVar_Kurtosis"] = grouped_pr["Kurtosis"].transform("std")

# Function to compute the variation ratio for RepVar_*
def variation_ratio(series):
    mean_val = series.mean()
    return ((series.std() / mean_val) * 100) if mean_val != 0 else 0

# Compute repetition-level variation (RepVar_*) per (Participant, Task)
grouped_pt = spectrumData.groupby(["Participant", "Task"])
spectrumData["RepVar_CentralGravity"] = grouped_pt["CentralGravity"].transform(variation_ratio)
spectrumData["RepVar_StandardDeviation"] = grouped_pt["StandardDeviation"].transform(variation_ratio)
spectrumData["RepVar_Skewness"] = grouped_pt["Skewness"].transform(variation_ratio)
spectrumData["RepVar_Kurtosis"] = grouped_pt["Kurtosis"].transform(variation_ratio)

# Print the final DataFrame to the console
print(spectrumData)
spectrumData.shape


# In[39]:


# Append repetition numbers directly to the Task column
spectrumData['Task'] = spectrumData['Task'] + (spectrumData.groupby('Task').cumcount() + 1).astype(str)

# Verify the updated DataFrame
print(spectrumData.head())


# In[40]:


output_file = output_dir + "Spectrum_Data_For_Spreadsheets.csv"
spectrumData.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


# # ---------------- Duration Extraction Function ---------------- #

# In[70]:


# DEFINE WHICH TIER TO ANALYZE
tier_index = 3  # 4th tier (indexing starts at 0)

# PREPARE DATA STORAGE
results = []

# Define a mapping for phonemes to specific task names
phoneme_to_task = {
    "DDKrate": "puh",  # Map all phonemes to specific tasks
}

# LOOP THROUGH EACH TEXTGRID FILE
for tg_path in textgrid_files:
    filename = os.path.basename(tg_path)
    base, ext = os.path.splitext(filename)
    
    # LOAD THE TEXTGRID
    try:
        tg = TextGrid.fromFile(tg_path)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue
    
    # VALIDATE TIER INDEX
    if tier_index >= len(tg.tiers):
        print(f"Tier index {tier_index} out of range for {filename}. Skipping.")
        continue
    
    tier = tg.tiers[tier_index]
    ddkrate_divisor = 9  # For full intervals
    
    # Track repetitions for each mapped task name
    task_counts = {
        "puh": 0,
        "tuh": 0,
        "kuh": 0,
    }
    
    # LOOP THROUGH EACH INTERVAL IN THE SPECIFIED TIER
    for interval in tier:
        thisPhoneme = interval.mark.strip()
        if thisPhoneme:  # Process non-empty labels
            # Map the phoneme to a task name
            task_name_base = phoneme_to_task.get(thisPhoneme, "puh")  # Default to "puh" if not mapped
            
            # Increment the repetition count for the task name
            task_counts[task_name_base] += 1
            
            # Create the task name with repetition number
            task_name = f"{task_name_base}_{task_counts[task_name_base]}"
            
            # Calculate duration and DDK rate
            duration = interval.maxTime - interval.minTime
            ddkrate = ddkrate_divisor / duration
            
            # Append result to the list
            results.append({
                "Participant": filename,
                "Task": task_name,
                "Duration": duration,
                "DDKRate": ddkrate
            })

# CONVERT RESULTS TO DATAFRAME
duration_df = pd.DataFrame(results)

# DISPLAY OR EXPORT THE DATAFRAME
print(duration_df.head())  # Preview the data
#output_csv = os.path.join(project_dir, "Duration_Data_For_R_2.csv")
#duration_df.to_csv(output_csv, index=False)  # Save to CSV if needed


# In[86]:


# Load the data from the original DataFrame
df = duration_df.copy()

# Define the sequence of tasks
task_sequence = ["puh", "tuh", "kuh"]

# Expand the DataFrame to replicate each row 9 times
df_expanded = df.loc[df.index.repeat(9)].reset_index(drop=True)

# Add an alternating Task column (puh, tuh, kuh)
df_expanded['Task'] = [
    f"{task}{i // 3 + 1}" for i, task in enumerate(task_sequence * (len(df_expanded) // 3))
]

# Convert all columns to string if needed
df_expanded = df_expanded.astype(str)

# Display the expanded DataFrame
print(df_expanded.shape)
print(df_expanded)


# In[87]:


output_file = output_dir + "Duration_Data_For_Spreadsheets.csv"
df_expanded.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


# # ---------------- Ratio Extraction Function ---------------- #

# In[79]:


# Initialize empty dataframe
ratio_data = pd.DataFrame(columns=["Participant", "Task", "Time", "F1", "F2", "Duration"])

# Loop through the TextGrid files
for file in textgrid_files:
    file_path = os.path.join(project_dir, file)
    sound_file = file_path.replace(".TextGrid", ".wav")
    
    try:
        tg = TextGrid.fromFile(file_path)
        snd = parselmouth.Sound(sound_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        continue

    tier_name = "vowel"  # Replace with your specific tier name
    tier = next((t for t in tg.tiers if t.name == tier_name), None)

    if tier is None:
        print(f"Tier '{tier_name}' not found in TextGrid {file}")
        continue

    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)

    new_rows = []
    for interval in tier.intervals:
        label = interval.mark.strip()
        start_time, end_time = interval.minTime, interval.maxTime
        if label:
            duration = end_time - start_time
            frame_count = int(duration / 0.0025)
            
            for frame in range(frame_count):
                frame_time = start_time + (frame * 0.0025)
                f1 = formant.get_value_at_time(1, frame_time)
                f2 = formant.get_value_at_time(2, frame_time)
                
                new_rows.append({
                    "Participant": "Normal_Hannah_DDK",
                    "Task": label,
                    "Time": frame_time,
                    "F1": f1,
                    "F2": f2,
                    "Duration": duration
                })
    
    ratio_data = pd.concat([ratio_data, pd.DataFrame(new_rows)], ignore_index=True)

# Processing ratio_data to compute BurstToBurst and DurationRatios
current_pt = ratio_data["Participant"].iloc[0]
previous_pt = ratio_data["Participant"].iloc[0]
current_task = ratio_data["Task"].iloc[0]
previous_task = ratio_data["Task"].iloc[0]
VOT_start = ratio_data["Time"].iloc[0]  
current_time = 0
previous_time = 0
table_row = 0

ratio_table = pd.DataFrame(columns=["Participant", "Rep", "Task", "BurstToBurst"])

kuh1gap = kuh2gap = 0

for index, row in ratio_data.iterrows():
    current_pt = row["Participant"]
    current_task = row["Task"]
    current_time = row["Time"]
    
    if current_task == "p2VOT" and previous_task == "kuh1":
        kuh1gap = current_time - previous_time
    if current_task == "p3VOT" and previous_task == "kuh2":
        kuh2gap = current_time - previous_time
    
    if current_task != previous_task and "VOT" in current_task and "p1VOT" not in current_task:
        burst_to_burst = current_time - VOT_start
        ratio_table.loc[table_row] = [previous_pt, "", previous_task, burst_to_burst]
        table_row += 1
        VOT_start = current_time

    if (index == len(ratio_data) - 1) or (current_task == "p1VOT" and previous_task == "kuh3"):
        if index == len(ratio_data) - 1:
            last_kuh = row["Time"]
            kuh_pt = current_pt
        else:
            last_kuh = previous_time
            kuh_pt = previous_pt
        burst_to_burst = last_kuh - VOT_start + (kuh1gap + kuh2gap) / 2
        ratio_table.loc[table_row] = [kuh_pt, "", "kuh3", burst_to_burst]
        table_row += 1
        VOT_start = current_time

    previous_pt = current_pt
    previous_task = current_task
    previous_time = current_time

ratio_table["BurstToBurst"] = ratio_table["BurstToBurst"].astype(float)
ratio_table["DurationRatio"] = None
for i in range(2, len(ratio_table), 3):
    duration_ratio = ratio_table.loc[i - 2, "BurstToBurst"] / ratio_table.loc[i - 1, "BurstToBurst"]
    ratio_table.loc[i - 2:i, "DurationRatio"] = duration_ratio

task_mapping = {"p1VOT": "puh1", "t1VOT": "tuh1", "k1VOT": "kuh1",
                "p2VOT": "puh2", "t2VOT": "tuh2", "k2VOT": "kuh2",
                "p3VOT": "puh3", "t3VOT": "tuh3", "k3VOT": "kuh3"}
ratio_table["Task"] = ratio_table["Task"].replace(task_mapping)

ratio_table["Rep"] = ratio_table["Task"].str.extract(r'(\d)').fillna("")
ratio_table["Task"] = ratio_table["Task"].str.replace(r'\d', '', regex=True)

for i in range(2, len(ratio_table), 3):
    ratio_table.iloc[i - 2:i + 1] = ratio_table.iloc[[i, i - 2, i - 1]].values

ratio_table["DistanceFrom1"] = ratio_table["DurationRatio"].astype(float) - 1

ratio_table = ratio_table[["Participant", "Rep", "Task", "BurstToBurst", "DurationRatio", "DistanceFrom1"]]

print(ratio_table)


# In[80]:


# Append repetition numbers directly to the Task column
ratio_table['Task'] = ratio_table['Task'] + (ratio_table.groupby('Task').cumcount() + 1).astype(str)

# Verify the updated DataFrame
print(ratio_table.head())


# In[81]:


output_file = output_dir + "Ratio_Data_For_Spreadsheets.csv"
ratio_table.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


# # ---------------- Gap Extraction Function ---------------- #

# In[82]:


# Initialize empty dataframe
gap_data = pd.DataFrame(columns=["Participant", "Task", "Time", "F1", "F2", "Duration"])

# Loop through the TextGrid files
for file in textgrid_files:
    file_path = os.path.join(project_dir, file)
    sound_file = file_path.replace(".TextGrid", ".wav")
    
    try:
        tg = TextGrid.fromFile(file_path)
        snd = parselmouth.Sound(sound_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        continue

    tier_name = "vowel"  # Replace with your specific tier name
    tier = next((t for t in tg.tiers if t.name == tier_name), None)

    if tier is None:
        print(f"Tier '{tier_name}' not found in TextGrid {file}")
        continue

    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000)

    new_rows = []
    for interval in tier.intervals:
        label = interval.mark.strip()
        start_time, end_time = interval.minTime, interval.maxTime
        if label:
            duration = end_time - start_time
            frame_count = int(duration / 0.0025)
            
            for frame in range(frame_count):
                frame_time = start_time + (frame * 0.0025)
                f1 = formant.get_value_at_time(1, frame_time)
                f2 = formant.get_value_at_time(2, frame_time)
                
                new_rows.append({
                    "Participant": "Normal_Hannah_DDK",
                    "Task": label,
                    "Time": frame_time,
                    "F1": f1,
                    "F2": f2,
                    "Duration": duration
                })
    
    gap_data = pd.concat([gap_data, pd.DataFrame(new_rows)], ignore_index=True)

# Processing gap_data to compute Gap durations
current_pt = gap_data["Participant"].iloc[0]
previous_pt = gap_data["Participant"].iloc[0]
current_task = gap_data["Task"].iloc[0]
previous_task = gap_data["Task"].iloc[0]
current_time = 0
previous_time = gap_data["Time"].iloc[0]
table_row = 0

gap_table = pd.DataFrame(columns=["Participant", "Task", "Gap"])

kuh1gap = kuh2gap = None

for index, row in gap_data.iterrows():
    current_pt = row["Participant"]
    current_task = row["Task"]
    current_time = row["Time"]
    
    if current_task == "p2vot" and previous_task == "kuh1":
        kuh1gap = current_time - previous_time
    if current_task == "p3vot" and previous_task == "kuh2":
        kuh2gap = current_time - previous_time
    
    if current_task != previous_task and "vot" in current_task.lower() and "p1vot" not in current_task.lower():
        gap = current_time - previous_time
        gap_table.loc[table_row] = [previous_pt, previous_task, gap]
        table_row += 1
    
    if index == len(gap_data) - 1 or (current_task == "p1vot" and previous_task == "kuh3"):
        kuh1gap = kuh1gap if kuh1gap else 0
        kuh2gap = kuh2gap if kuh2gap else 0
        gap = np.mean([kuh1gap, kuh2gap]) if kuh1gap and kuh2gap else 0
        gap_table.loc[table_row] = [current_pt, "kuh3", gap]
        table_row += 1

    previous_pt = current_pt
    previous_task = current_task
    previous_time = current_time

# Add Rep column
gap_table["Rep"] = gap_table["Task"].apply(lambda x: re.search(r"(\d)", x).group() if re.search(r"(\d)", x) else None)

# Remove numbers from task names
gap_table["Task"] = gap_table["Task"].str.replace(r'\d$', '', regex=True)

def reorder_rows(df):
    reordered = []
    for i in range(0, len(df), 3):
        if i + 2 < len(df):
            reordered.append(df.iloc[i + 2])
            reordered.append(df.iloc[i])
            reordered.append(df.iloc[i + 1])
    return pd.DataFrame(reordered)

gap_table = reorder_rows(gap_table)

print(gap_table)


# In[83]:


# Append repetition numbers directly to the Task column
gap_table['Task'] = gap_table['Task'] + (gap_table.groupby('Task').cumcount() + 1).astype(str)

# Verify the updated DataFrame
print(gap_table.head())


# In[84]:


output_file = output_dir + "GAP_Data_For_Spreadsheets.csv"
spectrumData.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")


# ## CONSOLIDATION

# In[88]:


# Define the relevant files for merging
files_to_merge = {
    "Formant": "FORMANT_Data_For_Spreadsheets.csv",
    "Spectrum": "Spectrum_Data_For_Spreadsheets.csv",
    "Duration": "Duration_Data_For_Spreadsheets.csv",
    "Gap": "GAP_Data_For_Spreadsheets.csv",
    "Ratio": "Ratio_Data_For_Spreadsheets.csv"
}

# Load datasets into a dictionary
dataframes = {}
for name, file in files_to_merge.items():
    file_path = os.path.join(output_dir, file)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Validate "Task" column
        if "Task" not in df.columns:
            raise ValueError(f"'Task' column not found in {name} dataset.")
        dataframes[name] = df
        print(f"Loaded {name} dataset with shape: {df.shape}")
    else:
        print(f"File not found: {file}")

# Ensure all datasets have the same unique "Task" values
common_tasks = set.intersection(*(set(df["Task"]) for df in dataframes.values()))
if len(common_tasks) != 9:
    raise ValueError(f"Inconsistent 'Task' values across datasets. Common tasks: {len(common_tasks)}")

# Filter datasets to only include common "Task" values
for name, df in dataframes.items():
    dataframes[name] = df[df["Task"].isin(common_tasks)]

# Merge datasets using "Task" as the key column
merged_df = dataframes["Formant"]  # Start with the Formant dataset
for name, df in dataframes.items():
    if name != "Formant":  # Skip the Formant dataset as it is already loaded
        merged_df = pd.merge(merged_df, df, on="Task", how="inner", suffixes=("", f"_{name}"))

# Display the resulting merged dataframe
print("Final merged dataset shape:", merged_df.shape)
print(merged_df.head())


# In[89]:


merged_df.shape


# In[2]:


jupyter nbconvert --to script acoustic_features.ipynb


# In[ ]:




