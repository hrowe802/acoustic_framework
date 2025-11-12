import pprint
import csv

with open("Prediction_Data_For_Python.csv") as source_csv:
	lines = source_csv.readlines()
print(lines)

# create empty list containing raw data rows
raw_data = []
participant = None
phoneme = None

# format the data into "participant", "phoneme", "spectrum"
for line in lines:
	if "starting script" in line or "pow" in line or len(line) < 2:
		continue
	if "new phoneme" in line:
		participant = line.split('|')[1]
		phoneme = line.split('|')[2].strip()
		continue
	raw_data.append([participant, phoneme, line.strip()])

# create list containing all possible phonemes
phonemes = ["p1", "t1", "k1", "p2", "t2", "k2", "p3", "t3", "k3"]

# create empty dictionary containing list of phonemes within each participant
full_dict = {}
for row in raw_data:
	participant = row[0]
	phoneme = row[1]
	spectrum = row[2]

	# initialize a new dictionary entry for the new participant
	if participant not in full_dict.keys():
		full_dict[participant] = {}
		for empty_phoneme in phonemes:
			full_dict[participant][empty_phoneme] = []
	full_dict[participant][phoneme].append(spectrum)

full_rows = []
for participant in full_dict.keys():
	for i in range(257):
		row = [participant]
		for phoneme in phonemes:
			row.append(full_dict[participant][phoneme][i])
		full_rows.append(row)

with open("/Users/hannahrowe/Desktop/Prediction_Data_For_R.csv", 'w') as output_file:
	header = ["Participant"] + phonemes
	writer = csv.writer(output_file)
	writer.writerow(header)
	writer.writerows(full_rows)
