import csv

# create function that separates task and rep
def output_row(csv_writer, row):
    row.insert(2, row[1][9])
    row[1] = row[1][0:9]
    csv_writer.writerow(row)
    print(row)

# open csv file
rows = []
with open("Cepstrum_Data_For_Python.csv") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        rows.append(row)

# create empty lists to append
header_line = rows[0]
rows = rows[1:]

with open("/Users/hannahrowe/Desktop/Cepstrum_Data_For_R.csv", 'w') as csvfile:
    csv_writer = csv.writer(csvfile)
    header_line = ["Participant", "Task", "Rep", "CPPS"]
    csv_writer.writerow(header_line)
    for row in rows:
        output_row(csv_writer, row)
