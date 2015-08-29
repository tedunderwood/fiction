# graphtagsovertime

import numpy as np
import csv, os, sys
from collections import Counter
import matplotlib.pyplot as plt
import SonicScrewdriver as utils

targetfile = input('Path to input file? ')

counts = dict()
alltags = set()
alldecades = set()
allcounts = Counter()

with open(targetfile, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        date = row['date']
        decade = 10 * int(int(date)/10)
        tagset = utils.get_tagset(row['genretags'])
        for tag in tagset:
            if tag == 'chirandom' and ('chiscifi' in tagset):
                continue
            if tag not in counts:
                counts[tag] = Counter()

            counts[tag][decade] += 1
            alltags.add(tag)
            alldecades.add(decade)
            allcounts[decade] += 1

sorted_decades = sorted(list(alldecades))
numdecs = len(sorted_decades)

colors = ['g-', 'b-', 'r-', 'k-', 'ro', 'go', 'bo', 'ko']

colindx = 0

for tag in alltags:
    if tag == 'crime' or tag == 'hardboiled' or tag == 'juvenile' or tag=='fantasy':
        continue
    print(tag + " : " + colors[colindx])
    x = np.array(sorted_decades)
    y = np.zeros(numdecs)
    for idx, dec in enumerate(sorted_decades):
        y[idx] = counts[tag][dec]

    plt.plot(x, y, colors[colindx])
    colindx += 1

plt.axis([1920,1980, 0, 32])
plt.show()








