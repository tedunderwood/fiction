#!/usr/bin/env python3

# main_experiment.py

# USAGE syntax:

# python3 main_experiment.py *command*

# Where *command* is one of the following codes.
# I have tried to start with functions that are
# relatively important to the argument of the article.

# sf_periods
# Runs sf_periods(), which simply assesses the ease
# of distinguishing science fiction from a random
# background in different periods.

# fantasy_periods
# Runs fantasy_periods(), which likewise assesses the
# ease of distinguishing fantasy from a random background.

# reliable_change
# Runs reliable_change_comparisons(), which supports a casual
# assertion I make in passing about the pace of change in
# science fiction.

# scar_19c
# Runs scarborough_to_19c_fantasy(), which compares
# a model based on Dorothy Scarborough's selection
# of "supernatural" fiction to books labeled "fantasy"
# by librarians.

# bailey_19c
# Runs bailey_to_19cSF(), which compares models based on J. O. Bailey's
# bibliography to models based on 19c works tagged as science
# fiction by postwar librarians.

# sfsurprise + *date*
#
# The syntax for this command differs from the general syntax
# above in requiring an additional argument, a date that will
# be the midpoint between the two periods, or strictly speaking
# the floor of the upper period. For instance, if you wanted to
# compare 1900-29 to 1930-59 you would say
#
# python3 main_experiment.py sfsurprise 1930
#
# The function runs get_rcc_surprise(date).
# This the function that produced the data
# ultimately used in figure 5.

# Note that not all of the functions below are directly
# used in the article. This script includes relics
# of earlier stages of research, and side branches
# to explore related questions.

import sys, os, csv, random
import numpy as np
import pandas as pd
import versatiletrainer2
import metaselector
from math import sqrt
import matplotlib.pyplot as plt

from scipy import stats

def add2dict(category, key, value):
    if key not in category:
        category[key] = []
    category[key].append(value)

def foldintodict(d1, d2, dictkey):
    for k, v in d1.items():
        if k not in d2:
            d2[k] = dict()
        d2[k][dictkey] = v

def divide_authdict(authdict, auths, ceiling, sizecap):
    random.shuffle(auths)
    part1 = []
    part2 = []

    for a in auths:
        lastauth = a
        if len(part1) > len(part2):
            part2.extend(authdict[a])
        else:
            part1.extend(authdict[a])

    if len(part1) >= sizecap and len(part2) >= sizecap:
        return part1, part2
    else:
        with open('errorlog.txt', mode = 'a', encoding = 'utf-8') as f:
            f.write('Error: imbalanced classes\n')
            f.write(str(ceiling) + '\t' + str(len(part1)) + '\t' + str(len(part2)) + '\t' + lastauth + '\n')

        if len(part1) > sizecap:
            part2.extend(part1[sizecap: ])
        elif len(part2) > sizecap:
            part1.extend(part2[sizecap: ])
        return part1, part2

def split_metadata(master, floor, ceiling, sizecap):
    '''
    This function serves quixotic_dead_end() and should
    probably be moved closer to it. It selects a chronological
    slice of master metadata and then divides that slice
    randomly into two partitions. Each partition is turned into
    two files: one that can be used for a model of SF vs
    mainstream lit, and one that can be used for a model of
    fantasy vs mainstream lit.
    '''

    dateslice = master[(master.firstpub >= floor) & (master.firstpub <= ceiling)]

    mainstream = dict()
    sf = dict()
    fant = dict()

    for idx in dateslice.index:
        tags = set(dateslice.loc[idx, 'tags'].split('|'))
        auth = dateslice.loc[idx, 'author']

        if 'juv' in tags:
            continue
            # no juvenile fiction included

        if 'random' in tags or 'randomB' in tags:
            add2dict(mainstream, auth, idx)
            continue

        if 'sf_loc' in tags or 'sf_oclc' in tags or 'sf_bailey' in tags:
            issf = True
        else:
            issf = False

        if 'fantasy_loc' in tags or 'fantasy_oclc' in tags or 'supernat' in tags:
            isfant = True
        else:
            isfant = False

        if issf and isfant:
            whim = random.choice([True, False])
            if whim:
                add2dict(sf, auth, idx)
            else:
                add2dict(fant, auth, idx)

        elif issf:
            add2dict(sf, auth, idx)

        elif isfant:
            add2dict(fant, auth, idx)

        else:
            pass
            # do precisely nothing

    sfauths = list(sf.keys())
    fantasyauths = list(fant.keys())
    mainstreamauths = list(mainstream.keys())

    maindocs1, maindocs2 = divide_authdict(mainstream, mainstreamauths, ceiling, sizecap)
    fantdocs1, fantdocs2 = divide_authdict(fant, fantasyauths, ceiling, sizecap)
    sfdocs1, sfdocs2 = divide_authdict(sf, sfauths, ceiling, sizecap)

    sf1 = master.loc[sfdocs1 + maindocs1]
    sf2 = master.loc[sfdocs2 + maindocs2]

    fant1 = master.loc[fantdocs1 + maindocs1]
    fant2 = master.loc[fantdocs2 + maindocs2]

    sf1.to_csv('../temp/sf1.csv')
    sf2.to_csv('../temp/sf2.csv')
    fant1.to_csv('../temp/fant1.csv')
    fant2.to_csv('../temp/fant2.csv')

def split_one_genre(master, floor, ceiling, positive_tags, genrename, sizecap):
    '''
    This function serves reliable_change_comparisons().
    '''

    dateslice = master[(master.firstpub >= floor) & (master.firstpub <= ceiling)]

    mainstream = dict()
    genre = dict()

    for idx in dateslice.index:
        tags = set(dateslice.loc[idx, 'tags'].split('|'))
        auth = dateslice.loc[idx, 'author']

        if 'juv' in tags:
            continue
            # no juvenile fiction included

        if 'random' in tags or 'randomB' in tags:
            add2dict(mainstream, auth, idx)
            continue

        for t in positive_tags:
            if t in tags:
                add2dict(genre, auth, idx)
                break

    genreauths = list(genre.keys())
    mainstreamauths = list(mainstream.keys())

    maindocs1, maindocs2 = divide_authdict(mainstream, mainstreamauths, ceiling, sizecap)
    genredocs1, genredocs2 = divide_authdict(genre, genreauths, ceiling, sizecap)

    partition1 = master.loc[genredocs1 + maindocs1]
    partition2 = master.loc[genredocs2 + maindocs2]

    partition1.to_csv('../temp/' + genrename + '1.csv')
    partition2.to_csv('../temp/' + genrename + '2.csv')

def fantasy_periods():
    '''
    Assesses the accuracy of models that distinguish fantasy from a random contrast set,
    in a series of periods defined by "periods". For the meaning of the parameters,
    consult versatiletrainer2.
    '''

    if not os.path.isfile('../results/fantasy_nojuv_periods.tsv'):
        with open('../results/fantasy_nojuv_periods.tsv', mode = 'w', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    sourcefolder = '../data/'
    metadatapath = '../metadata/mastermetadata.csv'
    tags4positive = {'fantasy_loc', 'fantasy_oclc'}
    tags4negative = {'random'}
    sizecap = 75

    periods = [(1800, 1899), (1900, 1919), (1920, 1949), (1950, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2010)]

    for excludebelow, excludeabove in periods:
        for i in range (5):

            name = 'fantasynojuv' + str(excludebelow) + 'to' + str(excludeabove) + 'v' + str(i)

            vocabpath = '../lexica/' + name + '.txt'
            metadatapath = ''

            metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, excludebelow = excludebelow, excludeabove = excludeabove, forbid4positive = {'juv'}, forbid4negative = {'juv'}, force_even_distribution = False)

            c_range = [.0003, .001, .006, .02, 0.1, 0.7, 3, 12]
            featurestart = 1000
            featureend = 6100
            featurestep = 200
            modelparams = 'logistic', 16, featurestart, featureend, featurestep, c_range

            matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

            meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))

            with open('../results/fantasy_nojuv_periods.tsv', mode = 'a', encoding = 'utf-8') as f:
                outline = name + '\t' + str(sizecap) + '\t' + str(excludebelow) + '\t' + str(excludeabove) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
                f.write(outline)

def sf_periods():

    '''
    Assesses the accuracy of models that distinguish SF from a random contrast set,
    in a series of periods defined by "periods". For the meaning of the parameters,
    consult versatiletrainer2.
    '''

    if not os.path.isfile('../results/sf_nojuv_periods.tsv'):
        with open('../results/sf_nojuv_periods.tsv', mode = 'w', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    sourcefolder = '../data/'
    metadatapath = '../metadata/mastermetadata.csv'
    tags4positive = {'sf_loc', 'sf_oclc'}
    tags4negative = {'random'}
    sizecap = 75

    periods = [(1800, 1899), (1900, 1919), (1920, 1949), (1950, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2010)]

    for excludebelow, excludeabove in periods:
        for i in range (5):

            name = 'sfnojuv' + str(excludebelow) + 'to' + str(excludeabove) + 'v' + str(i)

            vocabpath = '../lexica/' + name + '.txt'

            metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, excludebelow = excludebelow, excludeabove = excludeabove, forbid4positive = {'juv'}, forbid4negative = {'juv'})

            c_range = [.0003, .001, .006, .02, 0.1, 0.7, 3, 12]
            featurestart = 1000
            featureend = 6100
            featurestep = 200
            modelparams = 'logistic', 16, featurestart, featureend, featurestep, c_range

            matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

            meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))

            with open('../results/sf_nojuv_periods.tsv', mode = 'a', encoding = 'utf-8') as f:
                outline = name + '\t' + str(sizecap) + '\t' + str(excludebelow) + '\t' + str(excludeabove) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
                f.write(outline)

def accuracy(df, column):
    totalcount = len(df.realclass)
    tp = sum((df.realclass > 0.5) & (df[column] > 0.5))
    tn = sum((df.realclass <= 0.5) & (df[column] <= 0.5))
    fp = sum((df.realclass <= 0.5) & (df[column] > 0.5))
    fn = sum((df.realclass > 0.5) & (df[column] <= 0.5))
    assert totalcount == (tp + fp + tn + fn)

    return (tp + tn) / totalcount

def accuracy_loss(df):

    return accuracy(df, 'probability') - accuracy(df, 'alien_model')

def averagecorr(r1, r2):
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    themean = (z1 + z2) / 2
    return np.tanh(themean)

def write_a_row(r, outfile, columns):
    with open(outfile, mode = 'a', encoding = 'utf-8') as f:
        scribe = csv.DictWriter(f, fieldnames = columns, delimiter = '\t')
        scribe.writerow(r)

def sf2fantasy_divergence():

    ''' I suspect this is now deprecated.'''

    columns = ['testype', 'name1', 'name2', 'acc1', 'acc2', 'spearman', 'spear1on2', 'spear2on1', 'loss', 'loss1on2', 'loss2on1', 'meandate', 'ceiling', 'floor']

    outfile = '../results/sf2fantasynojuv.tsv'

    if not os.path.isfile(outfile):
        with open(outfile, mode = 'a', encoding = 'utf-8') as f:
            scribe = csv.DictWriter(f, delimiter = '\t', fieldnames = columns)
            scribe.writeheader()

    fantasymodels = pd.read_csv('../results/fantasy_nojuv_periods.tsv', sep = '\t')
    sfmodels = pd.read_csv('../results/sf_nojuv_periods.tsv', sep = '\t')

    # We group both sets by the "ceiling" columns, which is just a way of grouping models
    # that cover the same period. ("Meandate" is a better representation of central
    # tendency, but it can vary from one model to the next.)

    fantasy_grouped = fantasymodels.groupby('ceiling')
    sf_grouped = sfmodels.groupby('ceiling')

    fangroups = dict()
    sfgroups = dict()

    for ceiling, group in fantasy_grouped:
        fangroups[ceiling] = group

    for ceiling, group in sf_grouped:
        sfgroups[ceiling] = group

    for ceiling, fangroup in fangroups.items():
        for i1 in fangroup.index:

            # For each fantasy model in this ceiling group,
            # first, we test it against other fantasy models
            # in the same group.

            for i2 in fangroup.index:
                if i1 == i2:
                    continue
                    # we don't test a model against itself

                r = dict()
                r['testype'] = 'fantasyself'
                r['ceiling'] = ceiling
                r['floor'] = fangroup.loc[i1, 'floor']
                r['name1'] = fangroup.loc[i1, 'name']
                r['name2'] = fangroup.loc[i2, 'name']
                r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'] = get_divergence(r['name1'], r['name2'])
                r['acc1'] = fangroup.loc[i1, 'accuracy']
                r['acc2'] = fangroup.loc[i2, 'accuracy']
                r['meandate'] = (fangroup.loc[i1, 'meandate'] + fangroup.loc[i2, 'meandate']) / 2

                write_a_row(r, outfile, columns)

            sfgroup = sfgroups[ceiling]
            for idx in sfgroup.index:

                r = dict()
                r['testype'] = 'cross'
                r['ceiling'] = ceiling
                r['floor'] = fangroup.loc[i1, 'floor']
                r['name1'] = fangroup.loc[i1, 'name']
                r['name2'] = sfgroup.loc[idx, 'name']
                r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'] = get_divergence(r['name1'], r['name2'])
                r['acc1'] = fangroup.loc[i1, 'accuracy']
                r['acc2'] = sfgroup.loc[idx, 'accuracy']
                r['meandate'] = (fangroup.loc[i1, 'meandate'] + sfgroup.loc[idx, 'meandate']) / 2

                write_a_row(r, outfile, columns)

    # Now sf versus itself

    for ceiling, sfgroup in sfgroups.items():
        for i1 in sfgroup.index:

            # For each sf model in this ceiling group,
            # we test it against other sf models
            # in the same group.

            for i2 in sfgroup.index:
                if i1 == i2:
                    continue
                    # we don't test a model against itself

                r = dict()
                r['testype'] = 'sfself'
                r['ceiling'] = ceiling
                r['floor'] = sfgroup.loc[i1, 'floor']
                r['name1'] = sfgroup.loc[i1, 'name']
                r['name2'] = sfgroup.loc[i2, 'name']
                r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'] = get_divergence(r['name1'], r['name2'])
                r['acc1'] = sfgroup.loc[i1, 'accuracy']
                r['acc2'] = sfgroup.loc[i2, 'accuracy']
                r['meandate'] = (sfgroup.loc[i1, 'meandate'] + sfgroup.loc[i2, 'meandate']) / 2

                write_a_row(r, outfile, columns)

def sf_vs_fantasy():

    '''Not sure this was used for the current version of the article.'''

    sourcefolder = '../data/'
    metadatapath = '../metadata/mastermetadata.csv'
    tags4positive = {'fantasy_loc', 'fantasy_oclc', 'supernat'}
    tags4negative = {'sf_loc', 'sf_oclc', 'sf_bailey'}
    sizecap = 400

    excludebelow = 1800
    excludeabove = 3000


    name = 'fantasyvsSF3'

    vocabpath = '../lexica/' + name + '.txt'

    metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, excludebelow = excludebelow, excludeabove = excludeabove, forbid4positive = {'juv'}, forbid4negative = {'juv'}, numfeatures = 7000, force_even_distribution = True)

    # notice that I'm forbidding juvenile fiction on this run

    c_range = [.00005, .0001, .0003, .001]
    featurestart = 4000
    featureend = 5800
    featurestep = 200
    modelparams = 'logistic', 16, featurestart, featureend, featurestep, c_range

    matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

def sf_vs_fantasy_periods():

    '''Not sure this was used for the current version of the article.'''

    outresults = '../results/sf_vs_fantasy_periods2.tsv'

    if not os.path.isfile(outresults):
        with open(outresults, mode = 'w', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    sourcefolder = '../data/'
    metadatapath = '../metadata/mastermetadata.csv'
    tags4positive = {'fantasy_loc', 'fantasy_oclc'}
    tags4negative = {'sf_loc', 'sf_oclc'}
    sizecap = 70

    periods = [(1800, 1899), (1900, 1919), (1920, 1949), (1950, 1969), (1970, 1979), (1980, 1989), (1990, 1999), (2000, 2010)]

    for i in range (5):
        for excludebelow, excludeabove in periods:

            name = 'sf_vs_fantasy_periods2' + str(excludebelow) + 'to' + str(excludeabove) + 'v' + str(i)

            vocabpath = '../lexica/' + name + '.txt'

            metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, excludebelow = excludebelow, excludeabove = excludeabove, forbid4positive = {'juv'}, forbid4negative = {'juv'}, overlap_strategy = 'random')

            c_range = [.00001, .0001, .001, .01, 0.1, 1, 10, 100]
            featurestart = 1800
            featureend = 5500
            featurestep = 300
            modelparams = 'logistic', 16, featurestart, featureend, featurestep, c_range

            matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

            meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))

            with open(outresults, mode = 'a', encoding = 'utf-8') as f:
                outline = name + '\t' + str(sizecap) + '\t' + str(excludebelow) + '\t' + str(excludeabove) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
                f.write(outline)

def reliable_genre_comparisons():

    '''
    This function was used in the current version of the article.

    It addresses weaknesses in earlier versions of genre comparison
    by comparing only models *with no shared instances*.

    [Edit Jan 1: To be even more careful about leakage, make that
    *no shared authors.*]

    Doing that required a ----load of complexity I'm afraid. I have to first
    split each genre into disjoint sets, then create self-comparisons between
    those disjoint sets, as well as cross-comparisons between genres, and then
    finally compare the self-comparisons to the cross-comparisons.
    '''

    outmodels = '../results/reliable_models.tsv'
    outcomparisons = '../results/reliable_comparisons.tsv'
    columns = ['testype', 'name1', 'name2', 'ceiling', 'floor', 'meandate1', 'meandate2', 'acc1', 'acc2', 'alienacc1', 'alienacc2', 'spearman', 'spear1on2', 'spear2on1', 'loss', 'loss1on2', 'loss2on1']

    if not os.path.isfile(outcomparisons):
        with open(outcomparisons, mode = 'a', encoding = 'utf-8') as f:
            scribe = csv.DictWriter(f, delimiter = '\t', fieldnames = columns)
            scribe.writeheader()

    if not os.path.isfile(outmodels):
        with open(outmodels, mode = 'a', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    sourcefolder = '../data/'
    sizecap = 72

    c_range = [.00001, .0001, .001, .01, 0.1, 1, 10, 100]
    featurestart = 1500
    featureend = 6500
    featurestep = 300
    modelparams = 'logistic', 15, featurestart, featureend, featurestep, c_range

    master = pd.read_csv('../metadata/mastermetadata.csv', index_col = 'docid')
    periods = [(1800, 1909), (1880, 1924), (1900, 1949), (1910, 1959), (1930, 1969), (1950, 1979), (1970, 1989), (1980, 1999), (1990, 2010)]
    forbiddenwords = {'fantasy', 'fiction', 'science', 'horror'}

    # endpoints both inclusive

    for i in range(15):
        for floor, ceiling in periods:

            split_metadata(master, floor, ceiling, sizecap)

            # That function just above does the real work of preventing leakage,
            # by splitting the genre into two disjoint sets. This allows self-
            # comparisons that avoid shared authors, and are thus strictly
            # comparable to cross-comparisons.

            metaoptions = ['sf1', 'sf2', 'fant1', 'fant2']

            for m in metaoptions:
                metadatapath = '../temp/' + m + '.csv'
                vocabpath = '../lexica/' + m + '.txt'
                name = 'temp_' + m + str(ceiling) + '_' + str(i)

                if m == 'sf1' or m == 'sf2':
                    tags4positive = {'sf_loc', 'sf_oclc', 'sf_bailey'}
                else:
                    tags4positive = {'fantasy_loc', 'fantasy_oclc', 'supernat'}

                tags4negative = {'random', 'randomB'}

                metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, excludebelow = floor, excludeabove = ceiling, forbid4positive = {'juv'}, forbid4negative = {'juv'}, force_even_distribution = False, numfeatures = 6500, forbiddenwords = forbiddenwords)

                matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

                meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))

                with open(outmodels, mode = 'a', encoding = 'utf-8') as f:
                    outline = name + '\t' + str(sizecap) + '\t' + str(floor) + '\t' + str(ceiling) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
                    f.write(outline)

                os.remove(vocabpath)

            r = dict()
            r['testype'] = 'sfself'
            r['ceiling'] = ceiling
            r['floor'] = floor
            r['name1'] = 'temp_sf1' + str(ceiling) + '_' + str(i)
            r['name2'] = 'temp_sf2' + str(ceiling) + '_' + str(i)
            r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'], r['acc1'], r['acc2'], r['alienacc1'], r['alienacc2'], r['meandate1'], r['meandate2'] = get_divergence(r['name1'], r['name2'])
            write_a_row(r, outcomparisons, columns)

            r = dict()
            r['testype'] = 'fantasyself'
            r['ceiling'] = ceiling
            r['floor'] = floor
            r['name1'] = 'temp_fant1' + str(ceiling) + '_' + str(i)
            r['name2'] = 'temp_fant2' + str(ceiling) + '_' + str(i)
            r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'], r['acc1'], r['acc2'], r['alienacc1'], r['alienacc2'], r['meandate1'], r['meandate2'] = get_divergence(r['name1'], r['name2'])
            write_a_row(r, outcomparisons, columns)

            r = dict()
            r['testype'] = 'cross'
            r['ceiling'] = ceiling
            r['floor'] = floor
            r['name1'] = 'temp_sf1' + str(ceiling) + '_' + str(i)
            r['name2'] = 'temp_fant2' + str(ceiling) + '_' + str(i)
            r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'], r['acc1'], r['acc2'], r['alienacc1'], r['alienacc2'], r['meandate1'], r['meandate2'] = get_divergence(r['name1'], r['name2'])
            write_a_row(r, outcomparisons, columns)

            r = dict()
            r['testype'] = 'cross'
            r['ceiling'] = ceiling
            r['floor'] = floor
            r['name1'] = 'temp_sf2' + str(ceiling) + '_' + str(i)
            r['name2'] = 'temp_fant1' + str(ceiling) + '_' + str(i)
            r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'], r['acc1'], r['acc2'], r['alienacc1'], r['alienacc2'], r['meandate1'], r['meandate2'] = get_divergence(r['name1'], r['name2'])
            write_a_row(r, outcomparisons, columns)

def create_variant_models(modelname, tags4positive, tags4negative, splityear):
    '''
    Creates variant models that are then used by measure_parallax.
    '''

    outmodels = '../results/' + modelname + '_models.tsv'
    columns = ['testype', 'name1', 'name2', 'ceiling1', 'floor1', 'ceiling2', 'floor2', 'meandate1', 'meandate2', 'acc1', 'acc2', 'alienacc1', 'alienacc2', 'spearman', 'spear1on2', 'spear2on1', 'loss', 'loss1on2', 'loss2on1']

    if not os.path.isfile(outmodels):
        with open(outmodels, mode = 'a', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    sourcefolder = '../newdata/'
    sizecap = 75

    c_range = [.00001, .0001, .001, .01, 0.1, 1, 10, 100]
    featurestart = 1500
    featureend = 6500
    featurestep = 300
    modelparams = 'logistic', 15, featurestart, featureend, featurestep, c_range

    master = pd.read_csv('../meta/finalmeta.csv', index_col = 'docid')
    forbiddenwords = {}

    periods = [(1700, splityear - 1), (splityear, 2010)]

    for i in range(10):
        for floor, ceiling in periods:

            name = modelname + str(floor) + '_' + str(ceiling) + '_' + str(i)

            names = []

            names.append(name)

            metadatapath = '../meta/finalmeta.csv'
            vocabpath = '../lexica/' + name + '.txt'


            metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, extension = '.fic.tsv', excludebelow = floor, excludeabove = ceiling, forbid4positive = {'juv'}, forbid4negative = {'juv'}, force_even_distribution = False, numfeatures = 6500, forbiddenwords = forbiddenwords)

            matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

            meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))

            with open(outmodels, mode = 'a', encoding = 'utf-8') as f:
                outline = name + '\t' + str(sizecap) + '\t' + str(floor) + '\t' + str(ceiling) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
                f.write(outline)

            os.remove(vocabpath)

def measure_parallax(modelname, splityear):
    '''
    runs permuted comparisons between the models created by
    create_variant_models()
    '''

    periods = [(1700, splityear - 1), (splityear, 2010)]

    # identify the periods at issue

    outcomparisons = "../results/" + modelname + "_comparisons.tsv"
    columns = ['testype', 'name1', 'name2', 'ceiling1', 'floor1', 'ceiling2', 'floor2', 'meandate1', 'meandate2', 'acc1', 'acc2', 'alienacc1', 'alienacc2', 'spearman', 'spear1on2', 'spear2on1', 'loss', 'loss1on2', 'loss2on1']

    if not os.path.isfile(outcomparisons):
        with open(outcomparisons, mode = 'a', encoding = 'utf-8') as f:
            scribe = csv.DictWriter(f, delimiter = '\t', fieldnames = columns)
            scribe.writeheader()

    f0, c0 = periods[0]
    f1, c1 = periods[1]

    for i in range(3):
        for j in range(3):
                               
            name1 = modelname + str(f0) + '_' + str(c0) + '_' + str(i) 
            name2 = modelname + str(f1) + '_' + str(c1) + '_' + str(j)

            r = dict()
            r['testype'] = 'cross'
            r['ceiling1'] = c0
            r['floor1'] = f0
            r['ceiling2'] = c1
            r['floor2'] = f1
            r['name1'] = name1
            r['name2'] = name2
            r['spearman'], r['loss'], r['spear1on2'], r['spear2on1'], r['loss1on2'], r['loss2on1'], r['acc1'], r['acc2'], r['alienacc1'], r['alienacc2'], r['meandate1'], r['meandate2'] = get_divergence(r['name1'], r['name2'])

            write_a_row(r, outcomparisons, columns)

def loadforerror(dictionary, name):
    df = pd.read_csv('../modeloutput/' + name + '.csv', index_col = 'docid')
    for index, row in df.iterrows():

        if index not in dictionary:
            dictionary[index] = []
        dictionary[index].append(float(row.probability))


def errorbarfig3():

    f1 = 1910
    f2 = 1940
    c1 = 1939
    c2 = 1969

    oldperiod = dict()
    newperiod = dict()

    for i in range(5):
        for part in [1, 2]:

            name1 = 'rccsf'+ str(f1) + '_' + str(c1) + '_' + str(i) + '_' + str(part)
            name2 = 'rccsf'+ str(f2) + '_' + str(c2) + '_' + str(i) + '_' + str(part)
            loadforerror(oldperiod, name1)
            loadforerror(newperiod, name2)

    stderrors = []
    for key, value in oldperiod.items():
        if len(value) > 1:
            stderrors.append(np.std(value) / sqrt(len(value)))
    print('1910: ' + str(np.mean(stderrors)))
    olderror = np.mean(stderrors)

    stderrors = []
    for key, value in newperiod.items():
        if len(value) > 1:
            stderrors.append(np.std(value) / sqrt(len(value)))
    print('1940: ' + str(np.mean(stderrors)))
    newerror = np.mean(stderrors)

    print()
    print((olderror + newerror) / 2)

def get_divergence(sampleA, sampleB):
    '''
    This function applies model a to b, and vice versa, and returns
    a couple of measures of divergence: notably lost accuracy and
    z-tranformed spearman correlation.
    '''

    # We start by constructing the paths to the sampleA
    # standard model criteria (.pkl) and
    # model output (.csv) on the examples
    # originally used to train it.

    # We're going to try applying the sampleA standard
    # criteria to another model's output, and vice-
    # versa.

    model1 = '../modeloutput/' + sampleA + '.pkl'
    meta1 = '../modeloutput/' + sampleA + '.csv'

    # Now we construct paths to the test model
    # criteria (.pkl) and output (.csv).

    model2 = '../modeloutput/' + sampleB + '.pkl'
    meta2 = '../modeloutput/' + sampleB + '.csv'

    model1on2 = versatiletrainer2.apply_pickled_model(model1, '../newdata/', '.fic.tsv', meta2)
    model2on1 = versatiletrainer2.apply_pickled_model(model2, '../newdata/', '.fic.tsv', meta1)

    spearman1on2 = np.arctanh(stats.spearmanr(model1on2.probability, model1on2.alien_model)[0])
    spearman2on1 = np.arctanh(stats.spearmanr(model2on1.probability, model2on1.alien_model)[0])
    spearman = (spearman1on2 + spearman2on1) / 2

    loss1on2 = accuracy_loss(model1on2)
    loss2on1 = accuracy_loss(model2on1)
    loss = (loss1on2 + loss2on1) / 2

    alienacc2 = accuracy(model1on2, 'alien_model')
    alienacc1 = accuracy(model2on1, 'alien_model')

    acc2 = accuracy(model1on2, 'probability')
    acc1 = accuracy(model2on1, 'probability')

    meandate2 = np.mean(model1on2.std_date)
    meandate1 = np.mean(model2on1.std_date)

    return spearman, loss, spearman1on2, spearman2on1, loss1on2, loss2on1, acc1, acc2, alienacc1, alienacc2, meandate1, meandate2


## MAIN

args = sys.argv

command = args[1]

if command == "projectgothic":
    project_a_model('firsttest', {'lochorror', 'pbgothic', 'locghost', 'stangothic', 'chihorror'}, 
        {'random', 'chirandom'}, 1880)

elif command == "comparegothic":
    measure_parallax('firsttest', 1880)


