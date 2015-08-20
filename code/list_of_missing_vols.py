metafile = '/Users/tunder/Dropbox/fiction/meta/genremeta.csv'

tagstoget = ['grandom', 'pbgothic']

import SonicScrewdriver as utils
import csv, os, datetime

docidstoget = set()

with open(metafile, encoding = 'utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tagset = utils.get_tagset(row['genretags'])
        if 'drop' in tagset:
            continue
        getthis = False
        for tag in tagstoget:
            if tag in tagset:
                getthis = True

        if getthis:
            docidstoget.add(row['docid'])


filespresent = os.listdir('/Users/tunder/Dropbox/fiction/data/')

docidspresent = set([x.replace('.fic.tsv', '') for x in filespresent if x.endswith('.fic.tsv')])

docidsneeded = docidstoget - docidspresent

outfile = '/Users/tunder/Dropbox/fiction/meta/filestoget' + str(datetime.date.today()) + '.txt'
with open(outfile, mode = 'w', encoding = 'utf-8') as f:
    for docid in docidsneeded:
        outid = utils.dirty_pairtree(docid)
        f.write(outid + '\n')




