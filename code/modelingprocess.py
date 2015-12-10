# modelingprocess.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def remove_zerocols(trainingset, testset):
    ''' Remove all columns that sum to zero in the trainingset.
    '''

    columnsums = trainingset.sum(axis = 0)
    columnstokeep = []
    for i in range(len(columnsums)):
        if columnsums[i] > 0:
            columnstokeep.append(i)

    trainingset = trainingset.iloc[ : , columnstokeep]
    testset = testset.iloc[columnstokeep]

    return trainingset, testset

def sliceframe(dataframe, yvals, excludedrows, testrow):
    numrows = len(dataframe)
    newyvals = list(yvals)
    for i in excludedrows:
        del newyvals[i]
        # NB: This only works if we assume that excluded rows
        # has already been sorted in descending order !!!!!!!
        # otherwise indexes will slide around as you delete

    trainingset = dataframe.drop(dataframe.index[excludedrows])

    newyvals = np.array(newyvals)

    testset = dataframe.iloc[testrow]

    # Potential problem arises. What if some of these columns are
    # all zero, because the associated word occurs in none of the
    # documents still in the training matrix? An error will be
    # thrown. To avoid this, we remove columns that sum to zero.

    trainingset, testset = remove_zerocols(trainingset, testset)

    return trainingset, newyvals, testset

def normalizearray(featurearray, usedate):
    '''Normalizes an array by centering on means and
    scaling by standard deviations. Also returns the
    means and standard deviations for features.
    '''

    numinstances, numfeatures = featurearray.shape
    means = list()
    stdevs = list()
    lastcolumn = numfeatures - 1
    for featureidx in range(numfeatures):

        thiscolumn = featurearray.iloc[ : , featureidx]
        thismean = np.mean(thiscolumn)

        thisstdev = np.std(thiscolumn)

        if (not usedate) or featureidx != lastcolumn:
            # If we're using date we don't normalize the last column.
            means.append(thismean)
            stdevs.append(thisstdev)
            featurearray.iloc[ : , featureidx] = (thiscolumn - thismean) / thisstdev
        else:
            print('FLAG')
            means.append(thismean)
            thisstdev = 0.1
            stdevs.append(thisstdev)
            featurearray.iloc[ : , featureidx] = (thiscolumn - thismean) / thisstdev
            # We set a small stdev for date.

    return featurearray, means, stdevs

def model_one_volume(data5tuple):
    data, classvector, listtoexclude, i, usedate, regularization = data5tuple
    trainingset, yvals, testset = sliceframe(data, classvector, listtoexclude, i)
    newmodel = LogisticRegression(C = regularization)
    trainingset, means, stdevs = normalizearray(trainingset, usedate)
    newmodel.fit(trainingset, yvals)

    testset = (testset - means) / stdevs
    prediction = newmodel.predict_proba(testset)[0][1]
    if i % 50 == 0:
        print(i)
    # print(str(i) + "  -  " + str(len(listtoexclude)))
    return prediction
