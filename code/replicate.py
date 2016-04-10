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
        excludeif['negatives'] = set(negatives)
    # This is a way to exclude certain tags from the negative contrast set.

    return (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

def model_taglist(positive_tags, modelname):
    print('We are modeling these positive tags:')
    for tag in positive_tags:
        print(tag)

    sizecap = 1000
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath, vocabpath = paths

    exclusions = make_exclusions(0, 2000, sizecap, 'nonegatives')

    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))

    return allvolumes

def model_taglist_within_dates(positive_tags, modelname, mindate, maxdate):
    print('We are modeling these positive tags:')
    for tag in positive_tags:
        print(tag)

    sizecap = 1000
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath, vocabpath = paths

    exclusions = make_exclusions(mindate, maxdate, sizecap, 'nonegatives')

    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))

    return allvolumes

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

def project_tags(tagstoproject, tagtargets):

    targetstring = ','.join(tagtargets)
    projectstring = ','.join(tagstoproject)

    print('First we create a model of ' + targetstring)

    sizecap = 400

    modelname = targetstring + 'byitself'
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath1, vocabpath = paths

    exclusions = make_exclusions(0, 2000, sizecap, tagstoproject)
    # Note that we exclude tagstoproject from the negative contrast set, so the
    # contrast sets for the two models will be identical.

    positive_tags = tagtargets
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    print()
    print('Then we create a model of ' + projectstring + ' and use it to predict ' + targetstring)

    modelname = projectstring + 'predicts' + targetstring
    paths = make_paths(modelname)
    sourcefolder, extension, metadatapath, outputpath2, vocabpath = paths

    exclusions = make_exclusions(0, 2000, sizecap, 'nonegatives')

    positive_tags = list(tagtargets)
    positive_tags.extend(tagstoproject)
    testconditions = set(tagtargets)
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

def calibrate_detective():
    '''
    Tests accuracy of classification for detective fiction at different sample
    sizes.
    '''

    modelname = 'calibratedet'
    paths = make_paths(modelname)

    ## EXCLUSIONS.

    excludeif = dict()
    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['firstpub'] = 1700
    excludeabove['firstpub'] = 2020


    positive_tags = ['locdetective', 'locdetmyst', 'chimyst', 'locdetmyst', 'det100']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    sizes = [5,6,7,8,9,11,13,15,17,18,21,27,29,32,34,36,40,45,50,55,60,65,70,75,80,85,90,100]

    # with open('../results/collateddetectiveaccuracies.tsv', mode = 'a', encoding = 'utf-8') as f:
    #         f.write('sizecap\tavgsize\trawaccuracy\n')

    accuracies = []
    for sizecap in sizes:

        exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

        rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

        trainsizes = []
        for vol in allvolumes:
            trainsizes.append(vol[11])
            # this is unfortunately dependent on the exact way
            # logisticpredict formats its output

        avgsize = sum(trainsizes) / len(trainsizes)

        print(sizecap, avgsize, rawaccuracy)
        with open('../final/collateddetaccuracies.tsv', mode = 'a', encoding = 'utf-8') as f:
            f.write(str(sizecap) + '\t' + str(avgsize) + '\t' + str(rawaccuracy) + '\n')

    return None

def calibrate_stew():
    '''
    Tests accuracy of classification for ghastly stew at different sample
    sizes.
    '''

    modelname = 'calibratestew'
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

    positive_tags = ['stew']
    negative_tags = ['random', 'chirandom']
    testconditions = set()

    datetype = "firstpub"
    numfeatures = 10000
    regularization = .000075

    classifyconditions = (positive_tags, negative_tags, datetype, numfeatures, regularization, testconditions)

    sizes = [5,6,7,8,9,11,13,15,17,18,21,27,29,32,34,36,40,45,50,55,60,65,70,75,80,85,90,100]

    # with open('../results/collatedstewaccuracies.tsv', mode = 'a', encoding = 'utf-8') as f:
    #         f.write('sizecap\tavgsize\trawaccuracy\n')

    accuracies = []
    for sizecap in sizes:

        exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)

        rawaccuracy, allvolumes, coefficientuples = logisticpredict.create_model(paths, exclusions, classifyconditions)

        trainsizes = []
        for vol in allvolumes:
            trainsizes.append(vol[11])
            # this is unfortunately dependent on the exact way
            # logisticpredict formats its output

        avgsize = sum(trainsizes) / len(trainsizes)

        print(sizecap, avgsize, rawaccuracy)
        with open('../final/collatedstewaccuracies.tsv', mode = 'a', encoding = 'utf-8') as f:
            f.write(str(sizecap) + '\t' + str(avgsize) + '\t' + str(rawaccuracy) + '\n')

    return None

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
    command = ''

    if len(args) < 2:

        print('Your options include: ')
        print('  1) Model Indiana detective fiction by itself.')
        print('  2) Model LOC detective-esque categories by themselves.')
        print('  3) Model LOC and Indiana together.')
        print('  4) Extrapolate a model of LoC detective fiction to the Indiana exhibition.')
        print('  5) Extrapolate a model of detective fiction beyond a particular date.')
        print('  6) Extrapolate a model of one arbitrary genre tag to another.')
        print('  7) Extrapolate a model of gothic fiction beyond a particular date.')
        print('  8) Extrapolate a model of several tags to several others.')
        print('  9) Run detective prediction at many different sizes.')
        print('  10) Run ghastly stew prediction at many different sizes.')
        print('  11) Try to use detective fiction to predict scifi (fails).')
        print('  12) Model an arbitrary tag against random control set.')
        print('  13) Model all early gothic 1760-1840.')
        print('  14) Model all gothic.')
        print('  15) Model all SF.')

        userchoice = int(input('\nyour choice: '))

        if userchoice == 1:
            tagstomodel = ['det100']
            modelname = 'IndianaDetective'
            allvolumes = model_taglist(tagstomodel, modelname)
            print('Results are in allvolumes.')
        elif userchoice == 2:
            tagstomodel = ['locdetmyst', 'locdetective', 'chimyst']
            modelname = 'LOCdetective'
            allvolumes = model_taglist(tagstomodel, modelname)
            print('Results are in allvolumes.')
        elif userchoice == 3:
            tagstomodel = ['det100', 'locdetmyst', 'locdetective', 'chimyst']
            modelname = 'AllDetective'
            allvolumes = model_taglist(tagstomodel, modelname)
            print('Results are in allvolumes.')
        elif userchoice == 4:
            tagtoproject = ['locdetmyst', 'locdetective', 'chimyst']
            tagtarget = ['det100']
            project_tags(tagtoproject, tagtarget)
        elif userchoice == 5:
            command = 'extrapolate_detective_date'
            dividedate = int(input('date beyond which to project: '))
            project_detective_beyond_date(dividedate)
        elif userchoice == 6:
            tagtoproject = input('tag to project from: ')
            tagtarget = input('tag to project onto: ')
            project_tag_to_another(tagtoproject, tagtarget)
        elif userchoice == 7:
            dividedate = int(input('date beyond which to project: '))
            project_gothic_beyond_date(dividedate)
        elif userchoice == 8:
            tagstoproject = input('comma-separated list of tags to model and project from: ')
            tagstoproject = [x.strip() for x in tagstoproject.split(',')]
            tagtargets = input('comma-separated list of tags project onto: ')
            tagtargets = [x.strip() for x in tagtargets.split(',')]
            project_tags(tagstoproject, tagtargets)
        elif userchoice == 9:
            calibrate_detective()
        elif userchoice == 10:
            calibrate_stew()
        elif userchoice == 11:
            projectfrom = 'chimyst'
            projectonto = 'chiscifi'
            project_tag_to_another(projectfrom, projectonto)
        elif userchoice == 12:
            tagtomodel = input('tag to model (must be in metadata)? ')
            tagstomodel = [tagtomodel]
            allvolumes = model_taglist(tagstomodel, tagtomodel)
        elif userchoice == 13:
            tagstomodel = ['stangothic', 'pbgothic', 'lochorror', 'locghost']
            allvolumes = model_taglist_within_dates(tagstomodel, 'EarlyGothic', 1760, 1840)
        elif userchoice == 14:
            tagstomodel = ['stangothic', 'pbgothic', 'lochorror', 'locghost', 'chihorror']
            modelname = 'AllGothic'
            allvolumes = model_taglist(tagstomodel, modelname)
            print('Results are in allvolumes.')
        elif userchoice == 15:
            tagstomodel = ['locscifi', 'femscifi', 'anatscifi', 'chiscifi']
            modelname = 'AllSF'
            allvolumes = model_taglist(tagstomodel, modelname)
            print('Results are in allvolumes.')


    else:
        command = args[1]
        dividedate = 0

    print('Done.')





