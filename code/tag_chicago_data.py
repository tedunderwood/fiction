# tag_chicago_data.py

import csv, random, os
from collections import Counter
import SonicScrewdriver as utils

tagas = input("Tag volumes in this session as? ")

metadata = dict()
authordict = dict()
all_docids = set()

metasource = '/Users/tunder/Dropbox/US_Novel_Corpus/master_list_04-02-15_wgenres.csv'
with open(metasource, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        docid = row['BOOK_ID']
        row['date'] = row['PUBL_DATE']
        metadata[docid] = row
        authorstring = row['AUTH_LAST'] + ", " + row['AUTH_FIRST']
        row['author'] = authorstring
        authorstring = authorstring.replace(',', ' ')
        authorstring = authorstring.replace('.', ' ')
        authorwords = authorstring.lower().split()
        for word in authorwords:
            if word not in authordict:
                authordict[word] = set()

            authordict[word].add(docid)

        all_docids.add(docid)

existingmeta = dict()

metaout = '/Users/tunder/Dropbox/fiction/meta/chicagometa2.csv'
with open(metaout, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    outfieldnames = reader.fieldnames
    for row in reader:
        existingmeta[row['docid']] = row

def get_nationality(vol):
    if vol['NATIONALITY'].startswith('Amer'):
        nation = 'us'
    elif vol['NATIONALITY'].startswith('Brit'):
        nation = 'uk'
    elif vol['NATIONALITY'].startswith('Scot'):
        nation = 'uk'
    elif vol['NATIONALITY'].startswith('Cana'):
        nation = 'ca'
    elif vol['NATIONALITY'].startswith('Iris'):
        nation = 'ir'
    elif vol['NATIONALITY'].startswith('New Z'):
        nation = 'nz'
    elif vol['NATIONALITY'].startswith('Austra'):
        nation = 'au'
    else:
        nation = input('Author nationality? ')

    return nation

def search_author(authorwords):
    global all_docids, authordict, metadata, existingmeta
    has_all_names = set(all_docids)
    titlewords = set()
    for word in authorwords:
        word = word.lower()
        if word in authordict:
            hasthisword = authordict[word]
            print(len(hasthisword))
            has_all_names = has_all_names.intersection(hasthisword)
        elif word.startswith('_'):
            titlewords.add(word.replace('_', ''))

    if len(titlewords) > 0:
        filteredids = set()
        for docid in has_all_names:
            passes = True
            thistitle = metadata[docid]['TITLE'].lower().split()
            for word in titlewords:
                if word not in thistitle:
                    passes = False
            if passes:
                filteredids.add(docid)
        has_all_names = filteredids

    numberofmatches = len(has_all_names)

    if numberofmatches > 0 and numberofmatches < 500:
        for match in has_all_names:
            row = metadata[match]
            fields = [row['BOOK_ID'], str(row['date']), row['author'], row['TITLE']]
            outstring = " | ".join(fields)
            if row['BOOK_ID'] in existingmeta:
                outstring = outstring + " *have* " + existingmeta[row['BOOK_ID']]['genretags']
            print(outstring)

    elif numberofmatches < 1:
        print("No matches.")
    elif numberofmatches > 500:
        print("Too many matches.")
        print(numberofmatches)

def add_to_ficgenre(docid, existingmeta, tagas):
    global outfieldnames, metadata
    o = dict()
    j = metadata[docid]
    fields = [j['BOOK_ID'], str(j['date']), j['author'], j['TITLE']]
    print(" | ".join(fields))
    print(j['LC_GENRES'])
    o['docid'] = j['BOOK_ID']
    o['recordid'] = ''
    o['oclc'] = ''
    o['locnum'] = ''
    o['author'] = j['author']
    o['imprint'] = j['PUBL_CITY'] + ": " + j['PUBLISHER'] + ", " + j['PUBL_DATE']
    o['date'] = j['date']
    o['firstpub'] = j['date']
    if len(j['AUTH_DATES']) > 4:
        o['birthdate'] = int(j['AUTH_DATES'][0:4])
    else:
        o['birthdate'] = input('Author birth year? ')

    o['nationality'] = get_nationality(j)

    o['gender'] = input('Gender? ')
    o['title'] = j['TITLE']
    o['subjects'] = j['LC_GENRES'].replace('|', ';')
    o['enumcron'] = ''

    print("We will tag this as " + tagas)
    user = input('Anything to add to that? ')
    if len(user) > 2:
        o['genretags'] = tagas + ' | ' + user
    else:
         o['genretags'] = tagas

    existingmeta[j['BOOK_ID']] = o
    print('Done.')

def replace_existing(docid, existingmeta, tagas):
    existingmeta[docid]['genretags'] += " | " + tagas
    gender = input('Gender? ').strip()
    if gender == 'm' or gender == 'f':
        existingmeta[docid]['gender'] = gender
    else:
        print('Error.')
    print('Done.')

keepgoing = True

while keepgoing:

    userstring = input('? ')

    words = userstring.split()

    if words[0] == 's' and len(words) > 1:
        search_author(words[1:])

    elif words[0] == 'a' and len(words) == 2:
        doctoadd = words[1]
        if doctoadd in existingmeta:
            replace_existing(doctoadd, existingmeta, tagas)
        else:
            add_to_ficgenre(doctoadd, existingmeta, tagas)

    elif words[0] == 'quit' or words[0] == 'q' or words[0] == 'stop':
        keepgoing = False

    else:
        print("Didn't understand that. s to search, a to add, q to quit.")

# After editing existingmeta we write it back out
metaout = '/Users/tunder/Dropbox/fiction/meta/chicagometa2.csv'

allids = list()
for key, row in existingmeta.items():
    allids.append((row['date'], key))

allids.sort()
allids = [x[1] for x in allids]

with open(metaout, mode = 'w', encoding = 'utf-8') as f:
    writer = csv.DictWriter(f, fieldnames = outfieldnames)
    writer.writeheader()
    for anid in allids:
        row = existingmeta[anid]
        writer.writerow(row)

os.system("say 'The program is done.'")
