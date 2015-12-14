# entropyparallelizer.py

# Note: this script is no longer up-to-date
# as of December 14, 2015. If you want the
# up-to-date version, see poetic_entropy_parallelizer,
# which implements the chunking approach recommended
# by Yuancheng Zhu.

# This is the main module to run. Using
# parameters set at the bottom of the script,
# it calls a main_routine that gathers
# metadata for Chicago volumes and pairs
# bookids with paths to files.

# Then using multiprocessing, it distributes
# the work of calculating entropy over multiple
# cores by treating each file as a separate job.

# All the actual entropy calculations are done by
# functions in the entropyfunctions module.

# The basic idea of the whole project is strongly
# shaped by insights into literature and information
# theory from Mark Algee-Hewitt.

import os, csv, random
from multiprocessing import Pool
import entropyfunctions

def parallelize_over_paths(pathlist, number_of_cores):
    '''
    Calls get_volume_entropy for every path in pathlist
    and returns an ordered list of tuples that corresponds
    to the ordering of pathlist.
    '''
    pool = Pool(processes = number_of_cores)
    res = pool.map_async(entropyfunctions.get_volume_entropy, pathlist)

    res.wait()
    resultlist = res.get()
    pool.close()
    pool.join()

    return resultlist

def main_routine(number_of_cores, sourcedir, metadatapath, number_to_do, outputpath, windowradius):

    # I start by loading all the volumes in the metadata file as
    # dictionaries in a list. We're going to randomly select
    # number_to_do of these elements.

    metadata = []

    with open(metadatapath, encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)

    # If we have been told to select less than one file, then
    # select them ALL.

    if number_to_do == 0:
        number_to_do = len(metadata)

    selected = random.sample(metadata, number_to_do)

    # Now we need to align book_ids with paths to the
    # appropriate file. Our strategy is to iterate through
    # all paths but only take the ones that are present in a
    # set of selected ids. Let's now create that set.

    # In the process we'll also create a (new) metadata dictionary
    # which will be keyed by bookid, and contain as values the
    # row-dictionaries we read in from the metadata file.

    selectedids = set()
    metadata = dict()

    for row in selected:

        bookid = int(row['BOOK_ID'])
        selectedids.add(bookid)
        metadata[bookid] = row

    # Okay, now we're going to create a list of file paths that
    # aligns with a list of bookids.

    pathlist = []
    bookidlist = []

    # Make a list of all files in the source directory.
    sourcefiles = [x for x in os.listdir(sourcedir) if x.endswith('.txt')]

    # Now only take the paths that align with bookids in selectedids.
    for afile in sourcefiles:
        bookstring = afile.replace('.txt', '')
        if bookstring.isdigit():
            bookid = int(bookstring)
        # We are going to skip Matt's file 00010990a, because
        # I don't have it in my metadata. Also I like int(x) as a simple
        # way of stripping the leading zeroes in the Chicago filenames,
        # and that doesn't work for 00010990a.

        if bookid not in selectedids:
            continue
        else:
            bookidlist.append(bookid)
            filepath = sourcedir + afile
            twotuple = (filepath, windowradius)
            pathlist.append(twotuple)
            # In parallelizing the job we are going to pass
            # each worker a filepath plus the radius of the text
            # window to check. This will be the same for all files.

    # Now we actually get the entropies for all the files in the pathlist.
    # ALL THE REAL WORK HAPPENS HERE.

    results = parallelize_over_paths(pathlist, number_of_cores)

    output = []
    # we're going to create a list of dictionaries that
    # can be output easily as a csv

    for bookid, result in zip(bookidlist, results):

        # each result is a tuple
        relative_entropy, unigramct, typect, seqlen = result

        # for each volume we create a dictionary that
        # will be the output row

        o = dict()
        o['bookid'] = bookid
        o['date'] = metadata[bookid]['PUBL_DATE']
        o['author'] = metadata[bookid]['AUTH_LAST'] + ", " + metadata[bookid]['AUTH_FIRST']
        o['title'] = metadata[bookid]['TITLE']
        o['wordcount'] = unigramct
        o['wordsused'] = seqlen
        o['relent'] = relative_entropy
        o['typetoken'] = typect / unigramct
        o['libraries'] = metadata[bookid]['LIBRARIES']

        output.append(o)

    fields = ['bookid', 'date', 'wordcount', 'wordsused', 'libraries', 'relent', 'typetoken', 'title', 'author']
    with open(outputpath, mode = 'w', encoding = 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = fields)
        writer.writeheader()
        for outputdict in output:
            writer.writerow(outputdict)

if __name__ == '__main__':
    number_of_cores = 12
    windowradius = 4000
    # The halfwidth of the window of text to be checked, in words.
    # If you want to check the whole volume use something like
    # 10,000,000.

    # sourcedir = '/Users/tunder/Dropbox/CLEAN_TEXTS/'
    sourcedir = '/Volumes/TARDIS/US_Novel_Corpus/NOVELS_1880-1990/'
    metadatapath = '/Users/tunder/Dropbox/US_Novel_Corpus/master_list_04-02-15_wgenres.csv'
    number_to_do = 0
    outputpath = '/Users/tunder/discard/Chicago_relative_entropy.csv'

    main_routine(number_of_cores, sourcedir, metadatapath, number_to_do, outputpath, windowradius)
