# plot_mysteries_by_decade.py

import csv
from collections import Counter

allvols = Counter()
mystvols = Counter()

with open('/Users/tunder/Dropbox/US_Novel_Corpus/master_list_04-02-15_wgenres.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        date = int(row['PUBL_DATE'])
        genres = row['LC_GENRES'].split('|')

        allvols[date] += 1
        for genre in genres:
            if 'science fiction' in genre.lower() or "uthopia" in genre.lower():
                mystvols[date] += 1

x = list()
y = list()

for baseyear in range(1880,1990,10):
    ceiling = baseyear + 10
    alltotal = 0
    mysttotal = 0
    for i in range(baseyear, ceiling):
        alltotal += allvols[i]
        mysttotal += mystvols[i]
    x.append(baseyear)
    y.append(mysttotal/alltotal)
    print(baseyear)
    print(mysttotal/alltotal)
    print(mysttotal)
    print()




