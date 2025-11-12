import csv

# create function that changes VOT name to formant name
def vot_name_to_formant_name(vot_name):
    return vot_name[0] + "uh" + vot_name[1]

# open csv file
rows = []
with open("Formant_Data_For_Python.csv") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        rows.append(row)

# create empty lists to append
header_line = rows[0]
rows = rows[1:]
vot_rows = []
formant_rows = []

print("total data rows: {}".format(len(rows)))

# count number of VOT rows and number of formant rows
for row in rows:
    if "VOT" in row[1]:
        vot_rows.append(row)
    else:
        formant_rows.append(row)

print("number of formant rows: {}".format(len(formant_rows)))
print("number of vot rows: {}".format(len(vot_rows)))

# create column of VOT duration corresponding to ptk task (e.g. p1VOT goes next to puh1)
for formant_row in formant_rows:
    # find the matching vot row
    for vot_row in vot_rows:
        # if the participant matches and the translated vot name matches
        if formant_row[0] == vot_row[0] and formant_row[1] == vot_name_to_formant_name(vot_row[1]):
            formant_row.insert(2, vot_row[2])
    # take the task number from the task column (which is in the form puh2) and make it into its own column
    print(formant_row)
    formant_row.insert(2, formant_row[1][3])
    # reduce the task column to just the first three characters (puh)
    formant_row[1] = formant_row[1][:3]

# create new variables
header_line.insert(2, "VOT")
header_line.insert(2, "Rep")
print(header_line)
print(formant_rows)

# write to csv file
with open("/Users/hannahrowe/Desktop/Formant_Data_For_R.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(header_line)
    for row in formant_rows:
        csv_writer.writerow(row)
