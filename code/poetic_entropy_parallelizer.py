# entropyparallelizer.py

# This is the main module to run. Using
# parameters set at the bottom of the script,
# it calls a main_routine that gathers
# metadata for poetry volumes and pairs
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

def wordsplit(atext):
    punctuation = '.,():-—;"!?•$%@“”#<>+=/[]*^\'{}_■~\\|«»©&~`£·'
    atext = atext.replace('-', ' ')
    # we replace hyphens with spaces because it seems probable that for this purpose
    # we want to count hyphen-divided phrases as separate words 
    wordseq = [x.strip(punctuation).lower() for x in atext.split()]

    return wordseq

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

def main_routine(number_of_cores, sourcedir, metadatapath, number_to_do, outputpath):

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

    # MOST OF WHAT FOLLOWS IS SPECIFIC TO MY POETRY METADATA.
    # If you don't care about that, fast-forward down to
    # END METADATA MUNGING.

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
        bookid = row['docid']
        reception = row['recept']
        if reception == 'vulgar':
            row['recept'] = 0
        elif reception == 'elite':
            row['recept'] = 1

        if row['recept'] == 0 or row['recept'] == 1:
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
        bookid = afile.replace('.norm.txt', '')

        if bookid not in selectedids:
            continue
        else:
            bookidlist.append(bookid)
            filepath = sourcedir + afile
            pathlist.append(filepath)
            # For each volume, we pass a filepath and the radius
            # of a window of words to be considered. This will
            # be the same for all volumes.

    # END METADATA MUNGING.

    # Because we are going to calculate entropy and TTR on equal-sized chunks
    # for all the volumes, we need to know what the smallest volume size is.

    minsize = 1000000
    for filepath in pathlist:
        with open(filepath, encoding = 'utf-8') as f:
            filestring = f.read()
            words = wordsplit(filestring)
            # important to note, this is the same wordsplit function
            # that will be used in the entropy calculation

            wordlen = len(words)
            if wordlen < minsize:
                minsize = wordlen

    newpathlist = []
    for filepath in pathlist:
        newpathlist.append((filepath, minsize))
    pathlist = newpathlist

    print('The minimum doc size in words was ' + str(minsize))

    # Now we actually get the entropies for all the files in the pathlist.
    # REAL WORK HAPPENS HERE.

    results = parallelize_over_paths(pathlist, number_of_cores)

    output = []
    # we're going to create a list of dictionaries that
    # can be output easily as a csv

    # we also keep track of the maximum length of cumulative
    # sequences so we can output those in a separate matrix
    maxlen = 0

    for bookid, result in zip(bookidlist, results):

        # each result is a pentuple
        avg_conditional_ent, avg_pct_of_max, avg_TTR, cumulative_sequence, wordcount = result

        # for each volume we create a dictionary that
        # will be the output row

        o = dict()
        o['bookid'] = bookid
        o['date'] = metadata[bookid]['firstpub']
        o['wordcount'] = wordcount
        o['conditionalent'] = avg_conditional_ent
        o['pctofmaxent'] = avg_pct_of_max
        o['ttr'] = avg_TTR
        o['recept'] = metadata[bookid]['recept']
        o['author'] = metadata[bookid]['author']

        output.append(o)

        # we also keep track of max cumulative sequence length
        cumlen = len(cumulative_sequence)
        if cumlen > maxlen:
            maxlen = cumlen

    fields = ['bookid', 'date', 'wordcount', 'conditionalent', 'pctofmaxent', 'ttr', 'recept', 'author']
    with open(outputpath, mode = 'w', encoding = 'utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = fields)
        writer.writeheader()
        for outputdict in output:
            writer.writerow(outputdict)

    dimensions = ['ttr', 'conditional', 'normalized']
    outputmatrix = dict()
    for dimension in dimensions:
        outputmatrix[dimension] = []

    for bookid, result in zip(bookidlist, results):
        avg_conditional_ent, avg_pct_of_max, avg_TTR, cumulative_sequence, wordcount = result
        numchunks = len(cumulative_sequence)
        for dimension in dimensions:
            outputrow = []
            initialvalue = cumulative_sequence[0][dimension]
            for i in range(maxlen):
                if i < numchunks and initialvalue > 0:
                    relativetostart = cumulative_sequence[i][dimension] / initialvalue
                    outputrow.append(str(relativetostart))
                else:
                    outputrow.append('NA')
            outputmatrix[dimension].append(outputrow)

    for dimension in dimensions:
        filename = '/Users/tunder/discard/cumulative_' + dimension + '.csv'
        with open(filename, mode = 'w', encoding = 'utf-8') as f:
            writer = csv.writer(f)
            for row in outputmatrix[dimension]:
                writer.writerow(row)

if __name__ == '__main__':
    number_of_cores = 4

    # sourcedir = '/Users/tunder/Dropbox/CLEAN_TEXTS/'
    sourcedir = '/Users/tunder/Dropbox/GenreProject/python/reception/poetry/readable/'
    metadatapath = '/Users/tunder/Dropbox/GenreProject/python/reception/poetry/finalpoemeta.csv'
    number_to_do = 0
    outputpath = '/Users/tunder/discard/poetic_entropy3.csv'

    main_routine(number_of_cores, sourcedir, metadatapath, number_to_do, outputpath)
