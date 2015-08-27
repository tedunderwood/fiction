# get_chicago_scifi.py

import csv, random, os, shutil
from collections import Counter

nonvols = dict()
mystvols = dict()
metadata = dict()

fieldstowrite = ['docid', 'recordid', 'oclc', 'locnum', 'author', 'imprint', 'date', 'birthdate', 'firstpub', 'enumcron', 'subjects', 'title', 'nationality', 'gender', 'genretags']

categories = dict()
categories = [('chimyst', 'mystery', 15), ('chihorror', 'horror', 6), ('chifantasy', 'fantas', 5), ('chiscifi', 'science fiction', 15), ('chiutopia', 'utopia', 5), ('chirandom', '<NONSENSE>', 25)]

lastcategory = len(categories) - 1

# The way this works is that we test categories sequentially. The first element of each
# tuple is a label that will be applied if the second element is present in the subject/genre
# field. If we reach the end and none of the previous categories have been applied, the volume
# will be labeled as belonging to the random set.

# The third element of each tuple is the number of volumes to be
# randomly drawn in each decade from that category.

decades = [1920, 1930, 1940, 1950, 1960, 1970, 1980]
# These are the floors of the decades, so this reaches to 1989.

members = dict()
allmembers = dict()

for category in categories:
    catname = category[0]
    members[catname] = dict()
    allmembers[catname] = set()
    # Members is a dictionary broken down by decade.
    # Allmembers holds all members of that category for all decades.
    for dec in decades:
        members[catname][dec] = set()

with open('/Users/tunder/Dropbox/US_Novel_Corpus/master_list_04-02-15_wgenres.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        docid = row['BOOK_ID']
        metadata[docid] = row
        date = int(row['PUBL_DATE'])
        dec = int(date/10) * 10
        # Dec contains the floor year for this date's decade.

        if dec not in decades:
            continue
        # This volume is outside the chronological bounds of this sampling run.

        genres = row['LC_GENRES'].lower().split('|')

        placedspecifically = False

        for idx, category in enumerate(categories):

            catname = category[0]
            catsymptom = category[1]

            if idx == lastcategory:

                # This is the last category -- the random set. All books get added to this
                # category and have a chance of being selected. They may also be selected
                # for some other category.


                members[catname][dec].add(docid)
                allmembers[catname].add(docid)

            else:
                # Try all the genre tags in the genre string.
                for genre in genres:
                    if catsymptom in genre:
                        members[catname][dec].add(docid)
                        allmembers[catname].add(docid)
                        break
                        # We break because there's no reason to add a volume
                        # to the same set more than once.

# Now we have volumes in sets associated with genre categories.
# Volumes can belong to more than one category.
# We need to sample volumes and create lists of genre tags
# for each volume.

selectedvols = set()
randomvols = set()

for idx, category in enumerate(categories):
    catname = category[0]
    catN = category[2]

    for dec in decades:
        n = len(members[catname][dec])
        if catN < n:
            n = catN

        samp = random.sample(members[catname][dec], n)
        for docid in samp:
            selectedvols.add(docid)
            if idx == lastcategory:
                randomvols.add(docid)
                # We need to do this because there is otherwise no
                # way of knowing which of the selected vols were also
                # randomly selected.

# Selectedvols is a set because it doesn't matter which
# category you get selected *for*. We're always going
# to add all the tags for all the groups you belong to.
# Ran

volumetags = dict()
randomtag = categories[lastcategory][0]
for docid in selectedvols:
    if docid in randomvols:
        volumetags[docid] = {randomtag}
    else:
        volumetags[docid] = set()

    for cat in categories[: -1]:
        catname = cat[0]
        if docid in allmembers[catname]:
            volumetags[docid].add(catname)


outrows = list()

for anid in selectedvols:
    vol = metadata[anid]
    docid = vol['BOOK_ID']

    assert anid == docid

    zerostring = '0' * (8 - len(docid))
    id2write = zerostring + docid
    filename = '/Volumes/TARDIS/US_Novel_Corpus/NOVELS_1880-1990/' + id2write + '.txt'
    if not os.path.isfile(filename):
        print(vol['TITLE'] + " | " + vol['AUTH_LAST'] + " | " + vol['PUBL_DATE'] + " | " + vol['genretags'])
        continue

    outfilename = filename.replace('NOVELS_1880-1990', 'scifi')
    shutil.copyfile(filename, outfilename)

    o = dict()
    o['docid'] = id2write
    o['recordid'] = vol['LIBRARIES']
    o['oclc'] = ''
    o['locnum'] = ''
    o['title'] = vol['TITLE']
    o['author'] = vol['AUTH_LAST'] + ', ' + vol['AUTH_FIRST']
    if not o['author'].endswith('.'):
        o['author'] = o['author'] + ','
    if len(vol['AUTH_DATES']) > 4:
        o['birthdate'] = vol['AUTH_DATES'][0:4]
    else:
        o['birthdate'] = ''

    o['imprint'] = " ".join([vol['PUBL_CITY'], vol['PUBLISHER'], vol['PUBL_DATE']])
    o['firstpub'] = vol['PUBL_DATE']
    o['date'] = vol['PUBL_DATE']
    if vol['NATIONALITY'].startswith('Amer'):
        o['nationality'] = 'us'
    elif vol['NATIONALITY'].startswith('Brit'):
        o['nationality'] = 'uk'
    elif vol['NATIONALITY'].startswith('Scot'):
        o['nationality'] = 'uk'
    elif vol['NATIONALITY'].startswith('Cana'):
        o['nationality'] = 'ca'
    elif vol['NATIONALITY'].startswith('Iris'):
        o['nationality'] = 'ir'
    elif vol['NATIONALITY'].startswith('New Z'):
        o['nationality'] = 'nz'
    elif vol['NATIONALITY'].startswith('Austra'):
        o['nationality'] = 'au'
    else:
        o['nationality'] = '??'

    o['gender'] = ''
    o['genretags'] = ' | '.join(volumetags[docid])
    o['enumcron'] = ''
    o['subjects'] = vol['LC_GENRES'].replace('|', ';')

    outrows.append(o)

with open('/Users/tunder/Dropbox/fiction/meta/chicagometa.csv', mode='a', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = fieldstowrite)
    writer.writeheader()
    for row in outrows:
        writer.writerow(row)












