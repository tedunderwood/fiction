import logisticpredict, comparemodels
import datetime, sys

def ghastly_stew():

    ## PATHS.

    sourcefolder = '../newdata/'
    extension = '.fic.tsv'
    metadatapath = '../meta/finalmeta.csv'
    vocabpath = '../lexicon/new10k.csv'

    modelname = 'ghastlystew'
    outputpath = '../results/' + modelname + str(datetime.date.today()) + '.csv'

    # We can simply exclude volumes from consideration on the basis on any
    # metadata category we want, using the dictionaries defined below.

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['firstpub'] = 1700
    excludeabove['firstpub'] = 2020

    allstewgenres = {'cozy', 'hardboiled', 'det100', 'chimyst', 'locdetective', 'lockandkey', 'crime', 'locdetmyst', 'blcrime', 'anatscifi', 'locscifi', 'chiscifi', 'femscifi', 'stangothic', 'pbgothic', 'lochorror', 'chihorror', 'locghost'}

    # We have to explicitly exclude genres because the category "stew" in the
    # positive category wouldn't otherwise automatically exclude the constituent
    # tags that were used to create it.

    # I would just have put all those tags in the positive tag list, but then you'd lose
    # the ability to explicitly balance equal numbers of crime, gothic,
    # and science fiction, plus sensation novels. You'd get a list dominated by
    # the crime categories, which are better-represented in the dataset.

    excludeif['negatives'] = allstewgenres
    sizecap = 250

    # CLASSIFY CONDITIONS

    # We ask the user for a list of categories to be included in the positive
    # set, as well as a list for the negative set. Default for the negative set
    # is to include all the "random"ly selected categories. Note that random volumes
    # can also be tagged with various specific genre tags; they are included in the
    # negative set only if they lack tags from the positive set.

    positive_tags = ['stew']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    paths = (sourcefolder, extension, metadatapath, outputpath, vocabpath)
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)
    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))

def make_paths(modelname):
    '''
    Makes a pathtuple using a model name and a default set of
    paths to feature-vocab and metadata files.
    '''

    sourcefolder = '../newdata/'
    extension = '.fic.tsv'
    metadatapath = '../meta/finalmeta.csv'
    vocabpath = '../lexicon/new10k.csv'
    # These words will be used as features

    outputpath = '../results/' + modelname + str(datetime.date.today()) + '.csv'

    return (sourcefolder, extension, metadatapath, outputpath, vocabpath)

def make_exclusions(startdate, enddate, sizecap, negatives):
    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['firstpub'] = startdate
    excludeabove['firstpub'] = enddate

    if negatives != 'nonegatives':
        excludeif['negatives'] = negatives
    # This is a way to exclude certain tags from the negative contrast set.

    return (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

def project_detective_beyond_date(dividedate):

    print('First we create a model of detective fiction only after ' + str(dividedate))

    sizecap = 300

    modelname = 'detectivejustpost' + str(dividedate)
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath1, vocabpath = paths

    exclusions = make_exclusions(dividedate, 2000, sizecap, 'nonegatives')

    positive_tags = ['locdetective', 'locdetmyst', 'chimyst', 'det100']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    print()
    print('Then we create a model of detective fiction blindly predicting after ' + str(dividedate))

    modelname = 'detectivepredictpost' + str(dividedate)
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath2, vocabpath = paths

    exclusions = make_exclusions(0, 2001, sizecap, 'nonegatives')

    testconditions = {'1700', str(dividedate)}

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the second dataset at 0.5, accuracy is: ', str(rawaccuracy))
    print()

    # Now we compare the predictions made by these two models, comparing only
    # the volumes that are in both models but excluded from the training process
    # in the second model.

    comparemodels.compare_untrained(outputpath1, outputpath2)

def project_tag_to_another(tagtoproject, tagtarget):

    print('First we create a model of ' + tagtarget)

    sizecap = 400

    modelname = tagtarget + 'byitself'
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath1, vocabpath = paths

    exclusions = make_exclusions(0, 2000, sizecap, tagtoproject)
    # Note that we exclude tagtoproject from the negative contrast set, so the
    # contrast sets for the two models will be identical.

    positive_tags = [tagtarget]
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    print()
    print('Then we create a model of ' + tagtoproject + ' and use it to predict ' + tagtarget)

    modelname = tagtoproject + 'predicts' + tagtarget
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath2, vocabpath = paths

    exclusions = make_exclusions(0, 2001, sizecap, 'nonegatives')

    positive_tags = [tagtarget, tagtoproject]
    testconditions = {tagtarget}
    # That's the line that actually excludes tagtarget from training.

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the second dataset at 0.5, accuracy is: ', str(rawaccuracy))
    print()

    # Now we compare the predictions made by these two models, comparing only
    # the volumes that are in both models but excluded from the training process
    # in the second model.

    comparemodels.compare_untrained(outputpath1, outputpath2)

def the_red_and_the_black():

    sizecap = 140

    modelname = 'blackandthered'
    paths = make_paths(modelname)

    exclusions = make_exclusions(1700, 2001, sizecap, 'nonegatives')

    positive_tags = ['teamred']
    negative_tags = ['teamblack']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    accuracies = []
    for i in range(40):

        modelname = 'redandtheblack' + str(i)
        paths = make_paths(modelname)

        rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)
        print(rawaccuracy)
        accuracies.append(rawaccuracy)

    with open('finalaccuracies.csv', mode = 'w', encoding = 'utf-8') as f:
        for accuracy in accuracies:
            f.write(str(accuracy) + '\n')

def replicate_stew():

    sizecap = 140

    modelname = 'replicatestew'
    paths = make_paths(modelname)

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['firstpub'] = 1700
    excludeabove['firstpub'] = 2020

    allstewgenres = {'cozy', 'hardboiled', 'det100', 'chimyst', 'locdetective', 'lockandkey', 'crime', 'locdetmyst', 'blcrime', 'anatscifi', 'locscifi', 'chiscifi', 'femscifi', 'stangothic', 'pbgothic', 'lochorror', 'chihorror', 'locghost'}

    # We have to explicitly exclude genres because the category "stew" in the
    # positive category wouldn't otherwise automatically exclude the constituent
    # tags that were used to create it.

    # I would just have put all those tags in the positive tag list, but then you'd lose
    # the ability to explicitly balance equal numbers of crime, gothic,
    # and science fiction, plus sensation novels. You'd get a list dominated by
    # the crime categories, which are better-represented in the dataset.

    excludeif['negatives'] = allstewgenres
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

    positive_tags = ['stew']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    accuracies = []
    for i in range(20):

        rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)
        print(rawaccuracy)
        accuracies.append(rawaccuracy)

    with open('stewaccuracies.csv', mode = 'a', encoding = 'utf-8') as f:
        for accuracy in accuracies:
            f.write(str(accuracy) + '\n')

def replicate_detective():

    sizecap = 140

    modelname = 'replicatedet'
    paths = make_paths(modelname)

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['firstpub'] = 1700
    excludeabove['firstpub'] = 2020


    # We have to explicitly exclude genres because the category "stew" in the
    # positive category wouldn't otherwise automatically exclude the constituent
    # tags that were used to create it.

    # I would just have put all those tags in the positive tag list, but then you'd lose
    # the ability to explicitly balance equal numbers of crime, gothic,
    # and science fiction, plus sensation novels. You'd get a list dominated by
    # the crime categories, which are better-represented in the dataset.

    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

    positive_tags = ['locdetective', 'locdetmyst', 'chimyst', 'locdetmyst', 'det100']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    accuracies = []
    for i in range(20):

        rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)
        print(rawaccuracy)
        accuracies.append(rawaccuracy)

    with open('detaccuracies.csv', mode = 'a', encoding = 'utf-8') as f:
        for accuracy in accuracies:
            f.write(str(accuracy) + '\n')

def project_gothic_beyond_date(dividedate):

    print('First we create a model of gothic fiction only after ' + str(dividedate))

    sizecap = 300

    modelname = 'gothicjustpost' + str(dividedate)
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath1, vocabpath = paths

    exclusions = make_exclusions(dividedate, 2000, sizecap, 'nonegatives')

    positive_tags = ['lochorror', 'pbgothic', 'locghost', 'stangothic', 'chihorror']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    print()
    print('Then we create a model of gothic fiction blindly predicting after ' + str(dividedate))

    modelname = 'gothicpredictpost' + str(dividedate)
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath2, vocabpath = paths

    exclusions = make_exclusions(0, 2001, sizecap, 'nonegatives')

    testconditions = {'1700', str(dividedate)}

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the second dataset at 0.5, accuracy is: ', str(rawaccuracy))
    print()

    # Now we compare the predictions made by these two models, comparing only
    # the volumes that are in both models but excluded from the training process
    # in the second model.

    comparemodels.compare_untrained(outputpath1, outputpath2)

if __name__ == '__main__':

    args = sys.argv

    if len(args) < 2:

        print('Your options include: ')
        print('  1) Extrapolate a model of LoC "detective" fiction to the Indiana exhibition.')
        print('  2) Extrapolate a model of detective fiction beyond a particular date.')
        print('  3) Extrapolate a model of one tag to another.')
        print('  4) Extrapolate a model of gothic fiction beyond a particular date.')

        userchoice = int(input('\nyour choice: '))

        if userchoice == 1:
            command = 'extrapolate_indiana'
        elif userchoice == 2:
            command = 'extrapolate_detective_date'
            dividedate = int(input('date beyond which to project: '))
        elif userchoice == 3:
            command = 'extrapolate_tag'
        elif userchoice == 4:
            command = 'extrapolate_gothic_date'
            dividedate = int(input('date beyond which to project: '))

    else:
        command = args[1]
        dividedate = 0

    if command == 'extrapolate_detective_date':
        if dividedate == 0:
            dividedate = int(input('date beyond which to project: '))
        project_detective_beyond_date(dividedate)

    if command == 'extrapolate_tag':
        tagtoproject = input('tag to project: ')
        tagtarget = input('tag to model: ')
        project_tag_to_another(tagtoproject, tagtarget)

    if command == 'extrapolate_gothic_date':
        if dividedate == 0:
            dividedate = int(input('date beyond which to project: '))
        project_gothic_beyond_date(dividedate)









