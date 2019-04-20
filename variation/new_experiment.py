#!/usr/bin/env python3

# new_experiment.py

# USAGE syntax:

# python3 new_experiment.py *command*

# Where *command* is one of the following codes:
#    detectivevariations
#    sfvariations
#    gothicvariations
#
# The script also includes other functions to measure genre stability
# across time, but these are not used in the Critical Inquiry article
# I'm documenting.


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

def repeatedly_model(modelname, tags4positive, tags4negative, sizecap):

    outmodels = '../results/' + modelname + '_models.tsv'

    if not os.path.isfile(outmodels):
        with open(outmodels, mode = 'w', encoding = 'utf-8') as f:
            outline = 'name\tsize\tfloor\tceiling\tmeandate\taccuracy\tfeatures\tregularization\ti\n'
            f.write(outline)

    for i in range(100):
        name = modelname + str(i)
        sourcefolder = '../newdata/'
        metadatapath = '../meta/finalmeta.csv'
        vocabpath = '../lexica/' + name + '.txt'

        c_range = [.00001, .0001, .001, .01, 0.1, 1, 10, 100]
        featurestart = 800
        featureend = 6600
        featurestep = 200
        modelparams = 'logistic', 15, featurestart, featureend, featurestep, c_range
        forbiddenwords = {}
        floor = 1700
        ceiling = 2020

        metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches, vocablist = versatiletrainer2.get_simple_data(sourcefolder, metadatapath, vocabpath, tags4positive, tags4negative, sizecap, extension = '.fic.tsv', excludebelow = floor, excludeabove = ceiling,
            forbid4positive = {'juv'}, forbid4negative = {'juv'}, force_even_distribution = False, numfeatures = 6500, forbiddenwords = forbiddenwords)

        matrix, maxaccuracy, metadata, coefficientuples, features4max, best_regularization_coef = versatiletrainer2.tune_a_model(metadata, masterdata, classvector, classdictionary, orderedIDs, authormatches,
            vocablist, tags4positive, tags4negative, modelparams, name, '../modeloutput/' + name + '.csv')

        meandate = int(round(np.sum(metadata.firstpub) / len(metadata.firstpub)))
        floor = np.min(metadata.firstpub)
        ceiling = np.max(metadata.firstpub)

        with open(outmodels, mode = 'a', encoding = 'utf-8') as f:
            outline = name + '\t' + str(sizecap) + '\t' + str(floor) + '\t' + str(ceiling) + '\t' + str(meandate) + '\t' + str(maxaccuracy) + '\t' + str(features4max) + '\t' + str(best_regularization_coef) + '\t' + str(i) + '\n'
            f.write(outline)

        os.remove(vocabpath)

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
    featurestart = 1000
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

    for i in range(10):
        for j in range(10):

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
    create_variant_models('gothicvariants', {'lochorror', 'pbgothic', 'locghost', 'stangothic', 'chihorror'},
        {'random', 'chirandom'}, 1880)
elif command == "comparegothic":
    measure_parallax('gothicvariants', 1880)

elif command == "projectdetective":
    create_variant_models('detectivevariants', {'locdetective', 'locdetmyst', 'chimyst', 'det100'},
        {'random', 'chirandom'}, 1930)
elif command == "comparedetective":
    measure_parallax('detectivevariants', 1930)

elif command == "projectSF":
    create_variant_models('SFvariants', {'anatscifi', 'locscifi', 'chiscifi', 'femscifi'},
        {'random', 'chirandom'}, 1945)
elif command == "compareSF":
    measure_parallax('SFvariants', 1945)

elif command == "projectSF1930":
    create_variant_models('SFvariants1930', {'anatscifi', 'locscifi', 'chiscifi', 'femscifi'},
        {'random', 'chirandom'}, 1930)
elif command == "compareSF1930":
    measure_parallax('SFvariants1930', 1930)

elif command == 'detectivevariations':
    repeatedly_model('detectivemodels', {'locdetective', 'locdetmyst', 'chimyst', 'det100'},
        {'random', 'chirandom'}, 160)

elif command == 'sfvariations':
    repeatedly_model('scifi', {'anatscifi', 'locscifi', 'chiscifi', 'femscifi'},
        {'random', 'chirandom'}, 160)

elif command == 'gothicvariations':
    repeatedly_model('gothic', {'lochorror', 'pbgothic', 'locghost', 'stangothic', 'chihorror'},
        {'random', 'chirandom'}, 160)



