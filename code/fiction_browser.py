# fiction_browser

# Allows the user to search metadata with keywords
# and add particular volumes to the ficgenre
# metadata

import csv, os
from collections import Counter
import SonicScrewdriver as utils

tagas = input("Tag volumes in this session as? ")

metadata = dict()
authordict = dict()
all_docids = set()

metasource = '/Volumes/TARDIS/work/metadata/MergedMonographs.tsv'
with open(metasource, encoding = 'utf-8') as f:
    reader = csv.DictReader(f, delimiter = '\t')
    for row in reader:
        docid = row['HTid']
        row['date'] = utils.date_row(row)
        metadata[docid] = row
        authorstring = row['author']
        authorstring = authorstring.replace(',', ' ')
        authorstring = authorstring.replace('.', ' ')
        authorwords = authorstring.lower().split()
        for word in authorwords:
            if word not in authordict:
                authordict[word] = set()

            authordict[word].add(docid)

        all_docids.add(docid)

metaout = '/Users/tunder/Dropbox/fiction/meta/genremeta.csv'
with open(metaout, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    outfieldnames = reader.fieldnames


def search_author(authorwords):
    global all_docids, authordict, metadata
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
            thistitle = metadata[docid]['title'].lower().split()
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
            fields = [row['HTid'], str(row['date']), row['author'], row['title'], row['enumcron']]
            print(" | ".join(fields))
    elif numberofmatches < 1:
        print("No matches.")
    elif numberofmatches > 500:
        print("Too many matches.")
        print(numberofmatches)

def add_to_ficgenre(docid, existingfile, tagas):
    global outfieldnames, metadata
    with open(existingfile, mode = 'a', encoding = 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = outfieldnames)
        o = dict()
        j = metadata[docid]
        fields = [j['HTid'], str(j['date']), j['author'], j['title'], j['enumcron']]
        print(" | ".join(fields))
        o['docid'] = utils.clean_pairtree(j['HTid'])
        o['recordid'] = j['recordid']
        o['oclc'] = j['OCLC']
        o['locnum'] = j['LOCnum']
        o['author'] = j['author']
        o['imprint'] = j['imprint']
        o['date'] = j['date']
        o['firstpub'] = input('First publication date? ')
        o['birthdate'] = input('Author birth year? ')
        o['nationality'] = input('Nationality? ')
        o['gender'] = input('Gender? ')
        o['title'] = j['title']
        o['subjects'] = j['subjects']
        o['enumcron'] = j['enumcron']
        o['genretags'] = tagas
        for key, value in o.items():
            if o[key] == '<blank>':
                o[key] = ''
        writer.writerow(o)
    print('Done.')

keepgoing = True

while keepgoing:

    userstring = input('? ')

    words = userstring.split()

    if words[0] == 's' and len(words) > 1:
        search_author(words[1:])

    elif words[0] == 'a' and len(words) == 2:
        add_to_ficgenre(words[1], metaout, tagas)

    elif words[0] == 'quit' or words[0] == 'q' or words[0] == 'stop':
        keepgoing = False

    else:
        print("Didn't understand that. s to search, a to add, q to quit.")


os.system("say 'The program is done.'")
