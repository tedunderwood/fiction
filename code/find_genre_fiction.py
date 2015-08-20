# find_genre_fiction

# This is the script we initially used to populate
# a metadata table using only LoC headings.

import csv
from collections import Counter
import SonicScrewdriver as utils

ficids = set()

meta = dict()

ficsource = '/Volumes/TARDIS/work/fiction/metadata/fiction_metadata.csv'
with open(ficsource, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        htid = row['htid']
        dirtyhtid = utils.dirty_pairtree(htid)
        ficids.add(dirtyhtid)
        meta[dirtyhtid] = row

metasource = '/Volumes/TARDIS/work/metadata/MergedMonographs.tsv'

mysterysubjects = Counter()
scifisubjects = Counter()
gothsubjects = Counter()
gothclues = ['ghost stories', 'gothic revival', 'horror']
genretags = dict()

def add_tag(genretags, htid, tagtoadd):
    if htid not in genretags:
        genretags[htid] = set()

    if tagtoadd not in genretags[htid]:
        genretags[htid].add(tagtoadd)

selected = set()

with open(metasource, encoding = 'utf-8') as f:
    reader = csv.DictReader(f, delimiter = '\t')
    for row in reader:
        htid = row['HTid']
        subjects = row['subjects']
        subjlist = subjects.lower().split(';')
        if htid not in ficids and 'horror tales' not in subjlist:
            continue

        for subject in subjlist:
            if 'detective and mystery' in subject:
                add_tag(genretags, htid, 'locdetmyst')
                mysterysubjects[subject] += 1
                selected.add(htid)

            elif 'detective' in subject:
                add_tag(genretags, htid, 'locdetective')
                mysterysubjects[subject] += 1
                selected.add(htid)

            if 'science fiction' in subject:
                add_tag(genretags, htid, 'locscifi')
                scifisubjects[subject] += 1
                selected.add(htid)

            if 'ghost stories' in subject:
                add_tag(genretags, htid, 'locghost')
                gothsubjects[subject] += 1
                selected.add(htid)

            if 'horror' in subject:
                add_tag(genretags, htid, 'lochorror')
                gothsubjects[subject] += 1
                selected.add(htid)


def enumerate_tags(tagdict):
    for key, value in tagdict.items():
        print(key + ":   " + str(value))

enumerate_tags(mysterysubjects)
enumerate_tags(scifisubjects)
enumerate_tags(gothsubjects)

fieldnames.append('genretags')

with open('/Users/tunder/Dropbox/fiction/meta/ficmeta.csv', mode = 'w', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()
    for htid in selected:
        metarow = meta[htid]
        gts = '|'.join(genretags[htid])
        metarow['genretags'] = gts
        writer.writerow(metarow)




