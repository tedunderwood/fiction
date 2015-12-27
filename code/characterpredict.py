# characterpredict

# Trying to use logisticpredict on characters.

## PATHS.

import logisticpredict, comparemodels
import datetime

def generic():
    sourcefolder = '/Volumes/TARDIS/work/characterdata/charpredict/'
    extension = '.tsv'
    metadatapath = '/Users/tunder/Dropbox/character/meta/predictmeta.csv'
    vocabpath = '/Users/tunder/Dropbox/character/meta/predict1960-79vocab.csv'

    modelname = input('Name of model? ')
    outputpath = '/Users/tunder/Dropbox/character/results/' + modelname + str(datetime.date.today()) + '.csv'

    # We can simply exclude volumes from consideration on the basis on any
    # metadata category we want, using the dictionaries defined below.

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    daterange = input('Range of dates to use in the model? ')
    if ',' in daterange:
        dates = [int(x.strip()) for x in daterange.split(',')]
        dates.sort()
        if len(dates) == 2:
            assert dates[0] < dates[1]
            excludebelow['firstpub'] = dates[0]
            excludeabove['firstpub'] = dates[1]

    sizecap = 1000

    # CLASSIFY CONDITIONS

    # We ask the user for a list of categories to be included in the positive
    # set, as well as a list for the negative set. Default for the negative set
    # is to include all the "random"ly selected categories. Note that random volumes
    # can also be tagged with various specific genre tags; they are included in the
    # negative set only if they lack tags from the positive set.

    positive_tags = ['f']
    negative_tags = ['m']

    datetype = "firstpub"
    numfeatures = 1700
    regularization = .00011
    testconditions = set()

    paths = (sourcefolder, extension, metadatapath, outputpath, vocabpath)
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)
    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))

    tiltaccuracy = logisticpredict.diachronic_tilt(allvolumes, 'linear', [])

    print("Divided with a line fit to the data trend, it's ", str(tiltaccuracy))

def compare(dividedate):

    print('First we create a model of gender only after ' + str(dividedate))

    sizecap = 500

    modelname = 'post' + str(dividedate)
    sourcefolder = '/Volumes/TARDIS/work/characterdata/charpredict/'
    extension = '.tsv'
    metadatapath = '/Users/tunder/Dropbox/character/meta/predictmeta.csv'
    vocabpath = '/Users/tunder/Dropbox/character/meta/predictALLvocab.csv'
    outputpath1 = '/Users/tunder/Dropbox/character/results/' + modelname + str(datetime.date.today()) + '.csv'
    paths = (sourcefolder, extension, metadatapath, outputpath1, vocabpath)

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()
    excludebelow['firstpub'] = 1900
    excludeabove['firstpub'] = 1950
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

    positive_tags = ['f']
    negative_tags = ['m']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 2000
    regularization = .00009

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    print()
    print('Then we create a model of detective fiction blindly predicting after ' + str(dividedate))

    modelname = 'predictpost' + str(dividedate)
    outputpath2 = '/Users/tunder/Dropbox/character/results/' + modelname + str(datetime.date.today()) + '.csv'
    paths = (sourcefolder, extension, metadatapath, outputpath2, vocabpath)

    excludebelow['firstpub'] = 1780
    excludeabove['firstpub'] = 2000
    sizecap = 1000
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

    testconditions = {'1700', 1880}

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the second dataset at 0.5, accuracy is: ', str(rawaccuracy))
    print()

    # Now we compare the predictions made by these two models, comparing only
    # the volumes that are in both models but excluded from the training process
    # in the second model.

    comparemodels.compare_untrained(outputpath1, outputpath2)

if __name__ == '__main__':

    generic()
