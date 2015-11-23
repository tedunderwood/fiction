# metafilter.py

# Functions that process metadata for parallel_crossvalidate.py

import csv, random

knownnations = {'us', 'uk'}

# Obviously we know about other nations, but these are the
# one that need to match for us to count two volumes as
# 'perfect matches.' Requiring Germany to match would be
# unrealistic.

def dirty_pairtree(htid):
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

def get_metadata(classpath, volumeIDs, excludeif, excludeifnot, excludebelow, excludeabove):
    '''
    As the name would imply, this gets metadata matching a given set of volume
    IDs. It returns a dictionary containing only those volumes that were present
    both in metadata and in the data folder.

    It also accepts four dictionaries containing criteria that will exclude volumes
    from the modeling process.
    '''
    print(classpath)
    metadict = dict()

    with open(classpath, encoding = 'utf-8') as f:
        reader = csv.DictReader(f)

        anonctr = 0

        for row in reader:
            volid = row['docid']
            tagstring = row['genretags'].strip()
            taglist = tagstring.split('|')
            tagset = set([x.strip() for x in taglist])

            bail = False
            for key, value in excludeif.items():
                if row[key] == value:
                    bail = True
            for key, value in excludeifnot.items():
                if row[key] != value:
                    bail = True
            for key, value in excludebelow.items():
                if forceint(row[key]) < value:
                    bail = True
            for key, value in excludeabove.items():
                if forceint(row[key]) > value:
                    bail = True

            if bail:
                continue

            nation = row['nationality'].rstrip()

            if nation == 'ca':
                nation = 'us'
            elif nation == 'ir':
                nation = 'uk'
            # I hope none of my Canadian or Irish friends notice this.

            author = row['author'].strip()
            if len(author) < 1 or author == '<blank>':
                author = "anonymous" + str(anonctr)
                anonctr += 1

            metadict[volid] = dict()
            metadict[volid]['docid'] = volid
            metadict[volid]['pubdate'] = forceint(row['date'])
            metadict[volid]['birthdate'] = forceint(row['birthdate'])
            metadict[volid]['gender'] = row['gender'].rstrip()
            metadict[volid]['nation'] = nation
            metadict[volid]['author'] = author
            metadict[volid]['title'] = row['title']
            metadict[volid]['tagset'] = tagset
            metadict[volid]['firstpub'] = forceint(row['firstpub'])

    # We only return metadata entries for volumes that are also
    # in the list of volumeIDs -- ultimately extracted from the
    # filenames present in a data folder.

    allidsinmeta = set([x for x in metadict.keys()])
    allidsindir = set(volumeIDs)
    missinginmeta = len(allidsindir - allidsinmeta)
    missingindir = len(allidsinmeta - allidsindir)
    print("We have " + str(missinginmeta) + " volumes in missing in metadata, and")
    print(str(missingindir) + " volumes missing in the directory.")
    print(allidsinmeta - allidsindir)
    print('***')
    print(allidsindir - allidsinmeta)

    intersectiondict = dict()

    for anid in volumeIDs:
        if anid in metadict:
            intersectiondict[anid] = metadict[anid]

    return intersectiondict

def identify_class(negative_tags, positive_tags, docdict, categorytodivideon):
    ''' Given a string of genretags describing a volume,
    a group of tags describing the positive set,
    and another group describing the negative set,
    this function identifies the volume as either a member of
    the positive set, a member of the negative set, or
    a volume that for one reason or another should be
    dropped from the modeling process.

    categorytodivide on has a limited number of allowable values
    'tagset' -- divide based on presence/absence of tags
    or 'pubdate', 'firstpub', 'birthdate' -- divide based on date limits
    contained in the tags.
    '''

    positive = False
    negative = False

    tagset = docdict['tagset']

    # to remove spaces on either side of the virgule

    if 'drop' in tagset:
        return 'drop'

    if categorytodivideon == 'tagset':

        for tag in positive_tags:
            if tag in tagset:
                positive = True

        for tag in negative_tags:
            if tag in tagset and positive == False:
                negative = True
            elif tag in tagset and 'random' not in tag:
                negative = True

            # That bizarre little codicil means this:
            # generally we call any work with a negative tag "negative"
            # unless it was already tagged positive, and the only
            # thing making it negative is a "random" tag, which after all
            # is not incompatible with generic identity!!

    else:
        # in this case we assume that the category to divide on is
        # a date, and the targettags contain limits for the positive and
        # negative classes.

        posmin = positive_tags[0]
        posmax = positive_tags[1]
        negmin = negative_tags[0]
        negmax = negative_tags[1]

        thisdate = forceint(docdict[categorytodivideon])

        if thisdate >= posmin and thisdate <= posmax:
            positive = True
        elif thisdate >= negmin and thisdate <= negmax:
            negative = True

    if positive and negative:
        return 'drop'
    elif negative:
        return 'negative'
    elif positive:
        return 'positive'
    else:
        return 'drop'

def get_gender(avolume):
    if 'gender' in avolume:
        gender = avolume['gender']
    else:
        gender = ''

    return gender

def get_nationality(avolume):
    if 'nation' in avolume:
        nationality = avolume['nation']
    else:
        nationality = ''

    return nationality

def closest_idx(negative_volumes, positive_volume, datetype):
    '''
    Finds the volume in negative_volumes that most closely
    matches the date, nationality, and gender of positive_volume.
    Date is by far the most important category, but the function
    will reach one year away to get a better gender-nationality
    match if it can.
    '''

    global knownnations
    date = positive_volume[datetype]

    gender = get_gender(positive_volume)

    nationality = get_nationality(positive_volume)

    proximities = list()

    for atarget in negative_volumes:
        targetdate = atarget[datetype]
        proximity = abs(targetdate - date)
        targetgender = get_gender(atarget)
        targetnation = get_nationality(atarget)

        if gender != targetgender and gender != '' and targetgender != '':
            proximity += 0.6
        if nationality != targetnation and nationality in knownnations and targetnation in knownnations:
            proximity += 0.6

        # 0.6 is chosen to ensure that date is more important than either gender or nationality
        # separately, but not more important than both together. The algorithm will choose perfect
        # date-gender-nationality matches when available, but will prefer a perfect gender-nationality
        # match one year away to a complete failure on those criteria in the same year.

        proximities.append(proximity)

    closestidx = proximities.index(min(proximities))

    return closestidx

def label_classes(metadict, categorytodivideon, positive_tags, negative_tags, sizecap, datetype):
    ''' This takes as input the metadata dictionary generated
    by get_metadata. It subsets that dictionary into a
    positive class and a negative class. Instances that belong
    to neither class get ignored.

    categorytodivideon is either 'tagset', in which case we use positive_tags and
    negative_tags to identify vols in positive and negative classes,
    or it's some kind of date (pubdate, birthdate, firstpub), in which case
    positive_tags will be a pair of min and max dates for the positive class
    and negative_tags will be a min and max date for the negative class.
    '''

    all_instances = set([x for x in metadict.keys()])

    # The first stage is to find positive instances.

    all_positives = set()
    all_negatives = set()

    for docid, docdict in metadict.items():
        classflag = identify_class(negative_tags, positive_tags, docdict, categorytodivideon)
        if classflag == 'positive':
            all_positives.add(docid)
        elif classflag == 'negative':
            all_negatives.add(docid)

    if sizecap > 0 and len(all_positives) > sizecap:
        positives = random.sample(all_positives, sizecap)
    else:
        positives = list(all_positives)

    # If there's a sizecap we also want to ensure classes have
    # matching sizes and roughly equal distributions over time.
    # This is set up to assume that the negatives will be the
    # larger of the two groups, because usually in my process
    # the negative set is drawn from a large group of 'randomly
    # selected' volumes. Our goal is to match its distribution
    # as closely as possible, using the datetype we're matching
    # on (e.g. firstpub or birthdate) as well as gender and
    # nationality

    numpositives = len(all_positives)

    if sizecap > 0 and len(all_negatives) > numpositives:
        if categorytodivideon == 'tagset':
            negative_metadata = [metadict[x] for x in all_negatives]
            negatives = list()

            for anid in positives:
                this_positive = metadict[anid]

                closest_negative_idx = closest_idx(negative_metadata, this_positive, datetype)
                closest_negative = negative_metadata.pop(closest_negative_idx)
                # print(closest_negative)
                negatives.append(closest_negative['docid'])

                # print("MATCH this: " + str(this_positive['firstpub']) + " : " + this_positive['title'] + " " + this_positive['gender'])
                # print('with this: ' + str(closest_negative['firstpub']) + " : " + closest_negative['title'] + " " + closest_negative['gender'])
                # print()

        else:
            # if we're dividing classes by date, we obvs don't want to
            # ensure equal distributions over time.

            negatives = random.sample(all_negatives, sizecap)

    else:
        negatives = list(all_negatives)

    # Now we have two lists of ids.

    IDsToUse = set()
    classdictionary = dict()

    print()
    print("We have " + str(len(positives)) + " positive, and")
    print(str(len(negatives)) + " negative instances.")

    for anid in positives:
        IDsToUse.add(anid)
        classdictionary[anid] = 1

    for anid in negatives:
        IDsToUse.add(anid)
        classdictionary[anid] = 0

    return IDsToUse, classdictionary














