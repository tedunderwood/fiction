# considers multiple entropy csvs and creates a composite
# table using the normalized values most likely to be
# accurate

import csv
import numpy as np

globalset = dict()

with open ('/Users/tunder/discard/poetic_entropy780.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        globalset[row['bookid']] = row

specialset = dict()

with open ('/Users/tunder/discard/poetic_entropy2500.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        specialset[row['bookid']] = row

def adjust_dimension(dictionary, dimension):
    vector = []
    for key, value in dictionary.items():
        vector.append(float(value[dimension]))

    vector = np.array(vector)
    meanval = np.mean(vector)
    std = np.std(vector)

    for key, value in dictionary.items():
        value[dimension] = (float(value[dimension]) - meanval) / std


dimensions = ['ttr', 'pctofmaxent', 'conditionalent']

for dimension in dimensions:
    adjust_dimension(globalset, dimension)
    adjust_dimension(specialset, dimension)

for key, value in specialset.items():
    if int(value['wordcount']) < 2500:
        for dimension in dimensions:
            value[dimension] = globalset[key][dimension]

with open('/Users/tunder/discard/adjusted_poetic_entropy.csv', mode = 'w', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()
    for key, value in specialset.items():
        writer.writerow(value)



