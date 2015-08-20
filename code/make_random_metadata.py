import csv, random

sourcefolder = '/Users/tunder/Dropbox/fiction/meta/scifimeta.csv'

allrows = list()

with open(sourcefolder, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        tagtoadd = random.choice(['teamred', 'teamblue'])
        row['genretags'] = row['genretags'] + ' | ' + tagtoadd
        allrows.append(row)

outfolder = '/Users/tunder/Dropbox/fiction/meta/scifimeta2.csv'

with open(outfolder, mode='w', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()
    for row in allrows:
        writer.writerow(row)

