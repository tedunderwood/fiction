# metautils.py

# Utilities for munging metadata that are used in both
# logisticpredict and metafilter. Pulled out here so
# that we don't have to update them in both locations,
# creating potential inconsistency.

# right now, only infer_date is used

import sys

def infer_date(metadictentry, datetype):
    if datetype == 'pubdate':
        return metadictentry[datetype]
    elif datetype == 'firstpub':
        firstpub = metadictentry['firstpub']
        if firstpub > 1700 and firstpub < 2500:
            return firstpub
        else:
            return metadictentry['pubdate']
    else:
        print('Unsupported date type.')
        if datetype in metadictentry:
            return metadictentry[datetype]
        else:
            print('Fatal error in date type.')
            sys.exit(0)

def appendif(key, value, dictionary):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

def dirty_pairtree(htid):
    ''' Changes a 'clean' HathiTrust ID (with only chars that are
    legal in filenames) into a 'clean' version of the same name
    (which may contain illegal chars.)
    '''
    period = htid.find('.')
    prefix = htid[0:period]
    postfix = htid[(period+1): ]
    if '=' in postfix:
        postfix = postfix.replace('+',':')
        postfix = postfix.replace('=','/')
    dirtyname = prefix + "." + postfix
    return dirtyname

def forceint(astring):
    try:
        intval = int(astring)
    except:
        intval = 0

    return intval
