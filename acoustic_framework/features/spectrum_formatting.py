import csv

# create function that changes short name (p) to long name (puh)
def short_name_to_long_name(short_name):
    return short_name[0] + "uh"

# create function that separates task and rep
def output_row(csv_writer, row):
    row.insert(2, row[1][1])
    row[1] = short_name_to_long_name(row[1])
    csv_writer.writerow(row)
    print(row)

# open csv file
rows = []
with open("Spectrum_Data_For_Python.csv") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        rows.append(row)

# create empty lists to append
header_line = rows[0]
rows = rows[1:]

with open("/Users/hannahrowe/Desktop/Spectrum_Data_For_R.csv", 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    header_line = ["Participant", "Task", "Rep", "CentralGravity",
        "StandardDeviation", "Skewness", "Kurtosis"]
    csv_writer.writerow(header_line)

# for each task, reorder to kuh (* + 2), puh (* + 0), and tuh (* + 1)
    for task in range(int(len(rows)/9)):
        output_row(csv_writer, rows[task * 9 + 2])
        output_row(csv_writer, rows[task * 9 + 5])
        output_row(csv_writer, rows[task * 9 + 8])
        output_row(csv_writer, rows[task * 9 + 0])
        output_row(csv_writer, rows[task * 9 + 3])
        output_row(csv_writer, rows[task * 9 + 6])
        output_row(csv_writer, rows[task * 9 + 1])
        output_row(csv_writer, rows[task * 9 + 4])
        output_row(csv_writer, rows[task * 9 + 7])
