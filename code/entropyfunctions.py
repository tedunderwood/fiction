# entropyfunctions.py

# Functions that calculate conditional entropy and
# *relative* conditional entropy for a document.

# The basic idea of the whole project is
# shaped by insights into literature and information
# theory from Mark Algee-Hewitt, 2015.

# Conditional entropy = how much additional information
# would be required to represent the second
# word of each bigram if you already knew the first.
# If bigrams are very predictable, it will be low.

# Relative conditional entropy normalizes that score
# by dividing it by maximum entropy. In other words,
# given the length of the document and its distribution
# over words, what's the *maximum* conditional entropy
# it could have -- the entropy it would have if
# *no two bigrams repeated*?

from collections import Counter
from math import log
import random

def wordsplit(atext):
    punctuation = '.,():-—;"!?•$%@“”#<>+=/[]*^\'{}_■~\\|«»©&~`£·'
    atext = atext.replace('-', ' ')
    # we replace hyphens with spaces because it seems probable that for this purpose
    # we want to count hyphen-divided phrases as separate words 
    awordseq = [x.strip(punctuation).lower() for x in atext.split()]

    return awordseq

def get_volume_entropy(twotuple):
    '''
    Does what it says on the package. Note that functions to be
    parallelized can only have one argument. You have to unpack
    the tuple separately.
    '''
    apath, chunksize = twotuple

    with open(apath, encoding = 'utf-8') as f:
        text = f.read()
    avg_conditional_ent, avg_pct_of_max, avg_TTR, cumsequence, wordcount = measure_by_chunk(text, chunksize)

    return avg_conditional_ent, avg_pct_of_max, avg_TTR, cumsequence, wordcount

def measure_by_chunk(atext, chunksize):
    '''
    This function takes a document, represented as a single string, and returns
    its conditional entropy and TTR, calculated in several different ways.

    Since entropy and TTR are both ineradicably related to length, it calculates
    both quantities on chunks of a fixed size, which has been determined in advance
    as the size of the smallest document in the corpus. Then it averages across
    chunks.

    For each chunk, it calculates conditional entropy. In other words, it converts
    the string into a list of words, counts all the bigrams, and then figures
    out how much additional information would be required to represent the second
    word of each bigram if you already knew the first.

    Then it *normalizes* that, dividing by the maximum possible conditional entropy
    for this text, given its unigram distribution. This is estimated (not calculated
    exactly) by shuffling the text and recalculating conditional entropy.

    It also calculates type-token ratio for each chunk.

    Finally, in order to better understand relationships to length, we record all
    three of these values for a text that is cumulative (i.e., all chunks up to this
    one).
    '''

    wordseq = wordsplit(atext)

    # we count types and tokens in the full sequence

    overalltypect = len(set(wordseq))
    seqlen = len(wordseq)

    # Now we iterate through chunks
    overrun = False
    TTRlist = []
    conditional_entropy_list = []
    normalized_entropy_list = []
    cumulative_sequence = []

    for startposition in range(0, seqlen, chunksize):
        endposition = startposition + chunksize

        # If this (final) chunk would overrun the end of the sequence,
        # we adjust it so that it fits, and overlaps with the previous
        # chunk. Overlap is okay for calculation of mean TTR and entropy,
        # but the final chunk should not be used for calculation of
        # curves across the length of the text.

        if endposition >= seqlen:

            if endposition > seqlen:
                overrun = True

            endposition = seqlen
            startposition = endposition - chunksize

            if startposition < 0:
                print ('In at least one document, chunk size exceeds doc size.')
                # This is not okay, but let's proceed.
                startposition = 0

        thischunk = wordseq[startposition: endposition]

        TTR, conditional_entropy, normalized_entropy = get_all_measures(thischunk)

        TTRlist.append(TTR)
        conditional_entropy_list.append(conditional_entropy)
        normalized_entropy_list.append(normalized_entropy)

        if not overrun:
            cumulative_text = wordseq[0: endposition]
            cumTTR, cumconditional, cumnormalized = get_all_measures(cumulative_text)
            cumulative_dict = dict()
            cumulative_dict['ttr'] = cumTTR
            cumulative_dict['conditional'] = cumconditional
            cumulative_dict['normalized'] = cumnormalized
            cumulative_sequence.append(cumulative_dict)

    TTR = sum(TTRlist) / len(TTRlist)
    conditional_entropy = sum(conditional_entropy_list) / len(conditional_entropy_list)
    normalized_entropy = sum(normalized_entropy_list) / len(normalized_entropy_list)

    return conditional_entropy, normalized_entropy, TTR, cumulative_sequence, seqlen

def get_all_measures(awordseq):
    ''' Given a chunk of text, calculates TTR, conditional entropy,
    and a version of conditional entropy normalized by shuffling the
    words in the chunk.
    '''
    unigramdist, bigramdist, unigramct, bigramct, typect = get_distributions(awordseq)
    
    TTR = typect / len(awordseq)

    conditional_entropy, bigram_entropy, unigram_entropy = get_conditional_entropy(bigramdist, unigramdist, bigramct, unigramct)

    # Instead of actually calculating the maximum possible conditional
    # entropy for this set of words, we estimate it by the crude method
    # of shuffling the words randomly and recalculating conditional entropy
    # using the same function and the same information about unigrams.

    shuffle_entropy = get_shuffle_entropy(awordseq, unigramdist, unigramct, bigramct)

    normalized_entropy = conditional_entropy / shuffle_entropy

    return TTR, conditional_entropy, normalized_entropy

def get_distributions(awordseq):
    ''' Calculates distributions over bigrams and unigrams.
    '''

    bigramdist = Counter()
    unigramdist = Counter()
    unigramct = 0
    types = set()
    thislen = len(awordseq)

    for idx, word in enumerate(awordseq):

        unigramdist[word] += 1
        unigramct += 1
        types.add(word)

        # if this is the last word there is no nextword and
        # no bigram to be added!

        if idx > (thislen - 2):
            continue
        else:
            nextword = awordseq[idx + 1]
            bigramdist[(word, nextword)] += 1

    bigramct = unigramct - 1
    # this is the total number of bigrams in the corpus, understanding
    # bigrams as "tokens" not "types"

    typect = len(types)
    # this is the total number of unigram "types"

    return unigramdist, bigramdist, unigramct, bigramct, typect

def basic_entropy(distribution):
    entropy = 0

    for key, keyprob in distribution.items():
        entropy -= keyprob * log(keyprob, 2)

    return entropy

def get_conditional_entropy(bigramdist, unigramdist, bigramct, unigramct):
    '''
    Calculates conditional entropy. Given a joint
    distribution over bigrams and a distribution
    over unigrams, this calculates the conditional distribution of secondwords.
    '''

    # We start by normalizing the distributions so that they're probabilities;
    # they came to us as raw counts.

    for anykey in bigramdist.keys():
        bigramdist[anykey] = bigramdist[anykey] / bigramct

    for anykey in unigramdist.keys():
        unigramdist[anykey] = unigramdist[anykey] / unigramct

    # Now we calculate the entropies on these distributions separately.

    bigram_entropy = basic_entropy(bigramdist)
    unigram_entropy = basic_entropy(unigramdist)

    # According to the chain rule of entropy, the entropy of
    # bigrams conditional on unigrams is simply bigram entropy minus unigram entropy.

    conditional_entropy = bigram_entropy - unigram_entropy

    return conditional_entropy, bigram_entropy, unigram_entropy

def get_shuffle_entropy(oldseq, unigramdist, unigramct, bigramct):
    ''' Calculates max conditional entropy using existing
    information about the unigram distribution for a text,
    but *randomly shuffling* the words to produce a different
    bigram distribution. This gives us a rough estimate of
    what conditional entropy would look like with no meaningful
    dependence of secondwords on firstwords.
    '''

    awordseq = list(oldseq)
    # making a copy so we don't randomize the original sequence

    random.shuffle(awordseq)
    thislen = len(awordseq)
    newbigramdist = Counter()

    for idx, word in enumerate(awordseq):

        if idx > (thislen - 2):
            continue
        else:
            nextword = awordseq[idx + 1]
            newbigramdist[(word, nextword)] += 1

    shuffle_entropy, bigram_entropy, unigram_entropy = get_conditional_entropy(newbigramdist, unigramdist, bigramct, unigramct)

    return shuffle_entropy







