# select_random_corpus.py
#
# This module imports metadata about volumes
# in a given set of genre(s), as well as a
# given random set, and then helps the user
# select more volumes to balance the sets.
#
# It is loosely based on
#
# /Users/tunder/Dropbox/GenreProject/python/reception/select_poetry_corpus3.py


import csv, os, sys
import SonicScrewdriver as utils
import random

selecteddates = dict()
selected = list()
selectedmeta = dict()

knownnations = {'us', 'uk'}

def user_added_meta():
    meta = dict()
    meta['birthdate'] = input('Authors year of birth? ')
    meta['gender'] = input ('Authors gender? ')
    meta['nationality'] = input('Authors nationality? ')
    meta['firstpub'] = input('Date of first publication? ')
    return meta

def forceint(astring):
    try:
        intval = int(astring)
    except:
        intval = 0

    return intval

def get_metasets(inmetadata, targettags, randomtag):
    ''' Reads rows from path: inmetadata and identifies
    two groups of volumes: those that have one of the
    genre tag in the target group, and those that possess
    the "random" tag.
    '''
    randomvollist = list()
    targetvollist = list()

    with open(inmetadata, encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            firstpub = forceint(row['firstpub'])
            if firstpub < 1700 or firstpub > 2000:
                continue

            genretags = row['genretags'].split('|')
            genretags = set([x.strip() for x in genretags])
            # to remove spaces on either side of the virgule

            if 'drop' in genretags:
                continue

            target = False
            random = False
            for tag in targettags:
                if tag in genretags:
                    target = True
            if randomtag in genretags:
                random = True

            if random and target:
                # These should never happen at the same time.
                # A random tag should preclude other tags.
                print("ERROR CONDITION: random tag and target genre")
                print("tags both present for a single volume.")
                sys.exit(0)

            elif random:
                randomvollist.append(row)

            elif target:
                targetvollist.append(row)

    return targetvollist, randomvollist



def closest_idx(targetvollist, row):
    global knownnations
    date = forceint(row['firstpub'])
    gender = row['gender']
    nationality = row['nationality']

    proximities = list()

    for atarget in targetvollist:
        targetdate = forceint(atarget['firstpub'])
        proximity = abs(targetdate - date)
        targetgender = atarget['gender']
        targetnation = atarget['nationality']

        if gender != targetgender and gender != '' and targetgender != '':
            proximity += 0.1
        if nationality != targetnation and nationality in knownnations and targetnation in knownnations:
            proximity += 0.1

        proximities.append(proximity)

    closestidx = proximities.index(min(proximities))

    return closestidx

def get_difference(targetvollist, randomvollist):
    ''' Identifies volumes in targetvollist matching the dates of
    randomvollist and subtracts those from targetvollist in order to
    identify a list of dates that remain unmatched.
    '''

    if len(randomvollist) >= len(targetvollist):
        return []

    for row in randomvollist:
        closest_target = closest_idx(targetvollist, row)
        popped = targetvollist.pop(closest_target)
        print("MATCH this: " + str(row['firstpub']) + " : " + row['title'] + " " + row['gender'])
        print('with this: ' + str(popped['firstpub']) + " : " + popped['title'] + " " + popped['gender'])
        print()

    return targetvollist

# START MAIN PROCEDURE

fieldstocopy = ['recordid', 'oclc', 'locnum', 'author', 'imprint', 'enumcron', 'subjects', 'title']
fieldstowrite = ['docid', 'recordid', 'oclc', 'locnum', 'author', 'imprint', 'date', 'birthdate', 'firstpub', 'enumcron', 'subjects', 'title', 'nationality', 'gender', 'genretags']

sourcemetafile = "/Users/tunder/Dropbox/fiction/meta/genremeta.csv"

targetphrase = input("Comma-separated list of target genres: ")
targettags = [x.strip() for x in targetphrase.split(',')]

randomtag = input('Random tag to use for this run? ')

targetvollist, randomvollist = get_metasets(sourcemetafile, targettags, randomtag)

unmatchedtargets = get_difference(targetvollist, randomvollist)

usa = 0
nonusa = 0
male = 0
female = 0

for row in unmatchedtargets:
    gender = row['gender']
    nationality = row['nationality']
    if nationality == 'us':
        usa += 1
    else:
        nonusa += 1

    if gender == 'f':
        female += 1
    elif gender == 'm':
        male += 1


bydate = dict()
fictionmetadata = dict()
datesbydocid = dict()

with open('/Users/tunder/work/genre/metadata/ficmeta.csv', encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        docid = utils.clean_pairtree(row['htid'])
        fictionmetadata[docid] = row

        date = utils.date_row(row)
        datesbydocid[docid] = date
        if docid in selected:
            continue
        if date in bydate:
            bydate[date].append(docid)
        else:
            bydate[date] = [docid]

controlset = set()
controlmeta = dict()
usedfromselected = list()

print("IN UNMATCHED VOLUMES: ")
print("Male/female ratio: " + str(male) + " / " + str(female))
print("US / nonUS ratio: " + str(usa) + " / " + str(nonusa))

tarfemale = 0
confemale = 0
tarusa = 0
conusa = 0

for row in unmatchedtargets:
    print()
    print("Women in targetvols / women selected: " + str(tarfemale) + " / " + str(confemale))
    print("US in targetvols / US selected: " + str(tarusa)+ " / " + str(conusa) )

    theid = row['docid']
    date = forceint(row['firstpub'])
    usedfromselected.append(theid)
    print(theid)
    print(date)
    print(row['author'])
    print(row['title'])
    print(row['nationality'] + " -- " + row['gender'])
    if row['gender'].strip() == 'f':
        tarfemale += 1

    if row['nationality'] == 'us':
        tarusa += 1

    found = False
    while not found:
        candidates = bydate[date]
        choice = random.sample(candidates, 1)[0]
        print(choice)
        print(fictionmetadata[choice]['author'])
        print(fictionmetadata[choice]['title'])
        acceptable = input("ACCEPT? (y/n): ")
        if acceptable == "y":
            controlset.add(choice)
            found = True
            controlmeta[choice] = user_added_meta()
            controlmeta[choice]['docid'] = choice
            controlmeta[choice]['date'] = datesbydocid[choice]
            controlmeta[choice]['genretags'] = randomtag
            for field in fieldstocopy:
                controlmeta[choice][field] = fictionmetadata[choice][field]

            if controlmeta[choice]['gender'] == 'f':
                confemale += 1

            if controlmeta[choice]['nationality'] == 'us':
                conusa += 1

        if acceptable == 'quit':
            break
    if acceptable == 'quit':
        break

with open(sourcemetafile, mode='a', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = fieldstowrite)
    for docid, row in controlmeta.items():
        writer.writerow(row)







