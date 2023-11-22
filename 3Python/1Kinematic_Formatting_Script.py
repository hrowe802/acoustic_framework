#!/anaconda3/bin/python

import os
import re
import pprint
import csv
import code

'''
Read all of the .txt files exported from SMASH except for the header files and combine them into a single csv. These are a mix of single syllables (p1, t1, k1), repetitions (pataka), and full productions (pataka pataka pataka). This means all the data points are repeated three times, but segmented in three different ways. Add a "Time" column which continues to count upward from p1 to k3, regardless of segmentation. For example, the last timestamp in k1 is one time increment before the first timestamp in p2.
'''

data_directory = "/Users/hannahrowe/Google Drive/Research/Kinematic Files/Parsed (ALS)"
kinematic_timestamped_file = "/Users/hannahrowe/Google Drive/Research/Scripts/3 Python Scripts/Kinematic1_Data_For_Python.csv"
kinematic_duration_file = "/Users/hannahrowe/Google Drive/Research/Scripts/3 Python Scripts/Kinematic2_Data_For_Python.csv"

# if we assume that SMASH data exports use the same number of seconds between each data point, then we can use the same time increment throughout. replace the value below with the correct time increment
time_increment = 1

files = os.listdir(data_directory)

lines = []
for file in files:

    if "header" not in file:
        if "txt" in file:
            #print("\n----------------\n" + file + "\n----------------")
            with open(data_directory + "/" + file) as opened_file:
                file_text = opened_file.read()
                for line in file_text.split('\n'):
                    # start a new list for values in this row
                    this_line = []

                    # separate out columns that have one or more spaces (\s) between them
                    split_line = re.split("\s+", line.strip())

                    # ignore some lines which are blank
                    if len(split_line) != 4:
                        continue

                    # convert each value from scientific notation to a float number and add to the row
                    for value in split_line:
                        parsed_value = float(value)
                        this_line.append(parsed_value)

                    # split up everything before the ".txt"
                    participant_task_rep = file.split(".txt")[0]
                    # separate the participant name from the task_rep (which is in format like "t2", "rep2", or "full0")
                    participant, task_rep = participant_task_rep.split('_')

                    # set the rep to 0 if it is a full so there are no N/A values in the table at the end
                    if task_rep == 'full':
                        task_rep = 'full0'

                    # add the participant and task_rep to the line
                    this_line = this_line + [participant, task_rep]

                    # add the line to the table
                    lines.append(this_line)

pprint.pprint(lines)

# the lines should now be arranged in the format ['T1_d', 'T4_d', 'UL_LL_d', 'vT4_d', 'Participant', 'Task+Rep']
# they are not in order yet
# now that all lines are loaded into "lines", organize them into a dictionary called "time_series" with the format {"participant name": {"p1": [list of p1 values in order], "t1": [list of t1 values in order], (... continue through k3), then "rep1", "rep2", "rep3", then "full0"}

line_header = ['T1_d', 'T4_d', 'UL_LL_d', 'vT4_d', 'Participant', 'Task+Rep']
header_dictionary = {}
for index, header_name in enumerate(line_header):
    header_dictionary[header_name] = index

time_series = {}
for line in lines:
    participant = line[header_dictionary['Participant']]

    # if this is the first time seeing this participant, initialize an empty dictionary for them
    if participant not in time_series:
        time_series[participant] = {}
    task_rep = line[header_dictionary['Task+Rep']]

    # if this is the first time seeing this task_rep for this participant, initialize an empty list of data points
    if task_rep not in time_series[participant]:
        time_series[participant][task_rep] = []

    # actually add this data point to the correct list
    time_series[participant][task_rep].append(line)

# start a new table to track the durations of each task and rep
durations = []
# now that the time_series dictionary has all segments organized, we can loop through and add timestamps to each row
for participant in time_series:
    participant_time_series = time_series[participant]

    # first loop through the single syllable sequences
    syllable_sequence = ['p1', 't1', 'k1', 'p2', 't2', 'k2', 'p3', 't3', 'k3']
    t = 0
    for syllable in syllable_sequence:
        duration_start = t
        for line in participant_time_series[syllable]:
            line.insert(0, t)
            t = t + time_increment

        # find the time since starting this syllable
        duration = t - duration_start
        task = syllable[0]
        rep = syllable[1]
        key = participant + '|' + task + '|' + rep
        durations.append([key, "syllable", participant, task, rep, duration])

        # subtract a time increment at the end of each syllable because the value is repeated in the next syllable
        t = t - time_increment

        # print the first and last of each syllable to check time counting
        # print(participant_time_series[syllable][0])
        # print(participant_time_series[syllable][-1])

    # second loop through rep1, rep2, and rep3
    t = 0
    for rep in ['rep1', 'rep2', 'rep3']:
        duration_start = t
        for line in participant_time_series[rep]:
            line.insert(0, t)
            t = t + time_increment

        # find the time since starting this rep
        duration = t - duration_start
        task = "rep"
        rep = rep[-1]
        key = participant + '|' + task + '|' + rep
        durations.append([key, "rep", participant, task, rep, duration])

        # subtract a time increment at the end of each rep because the value is repeated in the next rep
        t = t - time_increment

        # print the first and last of each syllable to check time counting
        # print(participant_time_series[rep][0])
        # print(participant_time_series[rep][-1])

    # third loop through the full
    t = 0
    for line in participant_time_series['full0']:
        line.insert(0, t)
        t = t + time_increment

    # save the time at the end, which is the duration of the full
    duration = t
    task = "full"
    rep = "0"
    key = participant + '|' + task + '|' + rep
    durations.append([key, "full", participant, task, rep, duration])

    # print("single syllables go from {} to {}".format(participant_time_series['p1'][0][0], participant_time_series['k3'][-1][0]))
    # print("reps go from             {} to {}".format(participant_time_series['rep1'][0][0], participant_time_series['rep3'][-1][0]))
    # print("full goes from           {} to {}".format(participant_time_series['full0'][0][0], participant_time_series['full0'][-1][0]))

# now each row is in the format
pprint.pprint(time_series)
print("end time_series")

# save the full data set with one line per timestamp
with open(kinematic_timestamped_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time', 'T1_d', 'T4_d', 'UL_LL_d', 'vT4_d', 'Key', 'Participant', 'Task', 'Rep'])
    for participant in time_series:
        task_rep_sequence = ['p1', 't1', 'k1', 'p2', 't2', 'k2', 'p3', 't3', 'k3', 'rep1', 'rep2', 'rep3', 'full0']
        for task_rep in task_rep_sequence:
            for row in time_series[participant][task_rep]:
                # create the key in the format participant|task|rep
                task = task_rep[:-1]
                rep = task_rep[-1]
                key = participant + '|' + task + '|' + rep
                row = row[:5] + [key, participant, task, rep]
                csvwriter.writerow(row)

# save the duration table with 13 rows per participant (9 single-syllable durations, 3 reps, and a full)
with open(kinematic_duration_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Key', 'SequenceType', 'Participant', 'Task', 'Rep', 'Duration'])
    for row in durations:
        csvwriter.writerow(row)
