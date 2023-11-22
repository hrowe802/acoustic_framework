#!/anaconda3/bin/python

import os
import re
import pprint
import csv

'''
This script takes a formatted CSV with kinematic data and calculates the min and max of each column for each unique participant, task, and rep.

for each participant and each task


'''

kinematic_timestamped_file = "/Users/hannahrowe/Google Drive/Research/Scripts/3 Python Scripts/Kinematic1_Data_For_Python.csv"
kinematic_duration_file = "/Users/hannahrowe/Google Drive/Research/Scripts/3 Python Scripts/Kinematic2_Data_For_Python.csv"
output_csv_file_path = "/Users/hannahrowe/Google Drive/Research/Projects/VAL/VAL_Data (Kinematic).csv"

kinematic_timestamped_file_header = ['Time', 'T1_d', 'T4_d', 'UL_LL_d', 'vT4_d', 'Key', 'Participant', 'Task', 'Rep']
header_dictionary = {}
for index, header_name in enumerate(kinematic_timestamped_file_header):
    header_dictionary[header_name] = index
print("the indexes of the headers are: {}".format(header_dictionary))

markers = ['T1_d', 'T4_d', 'UL_LL_d', 'vT4_d']

# load the whole csv into a list of rows
rows = []
with open(kinematic_timestamped_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    # read the first row to compare to our hardcoded header in this script
    header_row = next(csvreader)
    print("the header row is {}".format(header_row))
    if header_row != kinematic_timestamped_file_header:
        print("We have a problem! The header row does not match the input_csv_header! This will fail!")
        exit(1)

    # read the rest of the rows after the header
    for row in csvreader:
        rows.append(row)

#### find the min and max for each unique participant+task+rep+marker

# ptrm is the unique identifier string which is short for participant+task+rep+marker, and pipes (|) are used to separate them. So for example it will be T1_d|MA041|t|1
min_max_ptrm = {}
ptrm_list = []

for row in rows:
    for marker in markers:
        # the csv_key should be something like MA041_t1
        csv_key = row[header_dictionary['Key']]
        row_time = row[header_dictionary['Time']]

        ptrm_name = marker + "|" + csv_key
        ptrm_value = float(row[header_dictionary[marker]])

        # if this is the first time we are seeing this ptrm then we need to establish it in the dictionary
        if ptrm_name not in min_max_ptrm:
            # initialize the first value to the min and the max
            min_max_ptrm[ptrm_name] = {"min_value": ptrm_value, "min_time": row_time, "max_value": ptrm_value, "max_time": row_time}
            ptrm_list.append(ptrm_name)

        # make this the new min if it is less than the old min
        if ptrm_value < min_max_ptrm[ptrm_name]['min_value']:
            min_max_ptrm[ptrm_name]['min_value'] = ptrm_value
            min_max_ptrm[ptrm_name]['min_time'] = row_time
        # make this the new max if it is more than the old max
        if ptrm_value > min_max_ptrm[ptrm_name]['max_value']:
            min_max_ptrm[ptrm_name]['max_value'] = ptrm_value
            min_max_ptrm[ptrm_name]['max_time'] = row_time
for ptrm in ptrm_list:
    pprint.pprint("{}: {}".format(ptrm, min_max_ptrm[ptrm]))

# now load the second csv file so we can extend each row to include min and max time and value for each marker
kinematic_duration_file_header = ['Key', 'SequenceType', 'Participant', 'Task', 'Rep', 'Duration']

header_dictionary = {}
for index, header_name in enumerate(kinematic_duration_file_header):
    header_dictionary[header_name] = index
print("the indexes of the headers are: {}".format(header_dictionary))

rows = []
with open(kinematic_duration_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    # read the first row to compare to our hardcoded header in this script
    header_row = next(csvreader)
    if header_row != kinematic_duration_file_header:
        print("We have a problem! The header row does not match the input_csv_header! This will fail!")
        exit(1)

    # read the rest of the rows after the header
    for row in csvreader:
        rows.append(row)

# now extend each row so they match the following format:
extended_header = ['Key', 'SequenceType', 'Participant', 'Task', 'Rep', 'Duration',
                   'T1_d_min_value',    'T1_d_min_time',    'T1_d_max_value',    'T1_d_max_time',
                   'T4_d_min_value',    'T4_d_min_time',    'T4_d_max_value',    'T4_d_max_time',
                   'UL_LL_d_min_value', 'UL_LL_d_min_time', 'UL_LL_d_max_value', 'UL_LL_d_max_time',
                   'vT4_d_min_value',   'vT4_d_min_time',   'vT4_d_max_value',    'vT4_d_max_time']

extended_rows = []
for row in rows:
    row_key = row[0]
    # the first 6 columns are the same, so copy them over from the original row to the exended row
    extended_row = row.copy()
    for marker in markers:
        ptrm = marker + "|" + row_key
        if ptrm not in min_max_ptrm:
            print("We have a problem! The ptrm ({}) is missing from the min and max data! This will fail!".format(ptrm))
            exit(1)
        this_min_max = min_max_ptrm[ptrm]
        min_max_keys = ['min_value', 'min_time', 'max_value', 'max_time']
        for min_max_key in min_max_keys:
            if min_max_key not in this_min_max:
                print("We have a problem! The min_max_key ({}) is missing from the min_max_data for {}! This will fail!".format(min_max_key, ptrm))
                exit(1)
            # actually add the min max values and times for this marker to the row
            extended_row.append(this_min_max[min_max_key])
    extended_rows.append(extended_row)

pprint.pprint(extended_rows)

with open(output_csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(extended_header)
    for row in extended_rows:
        csvwriter.writerow(row)
