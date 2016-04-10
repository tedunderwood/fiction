fiction/code
============

replicate.py
------------
The central script that you call to reproduce tests from the article. If you run it interactively, it will query you and ask you which part of the project you want to reproduce, listing options roughly in article order. You can also specify other settings to compare other sets of genres.

The comments included in replicate.py explain a lot of otherwise-obscure aspects of the code, especially the parameters that get passed to logisticpredict in order to tell it how to construct a negative contrast set, and possibly a "donottrain" or "test-only" set that gets predicted but *never* used for training. (This is beyond the ordinary "test set" associated with leave-one-out crossvalidation.) These are the features that allow the script to programmatically "extrapolate" a model from one genre to another, but it is frankly an incredible pain to code -- not because it is at bottom conceptually complex, but because it creates a whole lot of minor loose ends and slight variations. I've tried to explain them in the comments at the start of replicate.py.

logisticpredict.py
------------------
The main modeling script, called by replicate.py with specific parameters. Does leave-one-out crossvalidation by author. If you run this by itself it allows you to specify a set of positive and negative tags for a model.

modelingprocess.py
------------------
Actually does the modeling for each pass of leave-one-out crossvalidation.

metafilter.py
-------------
Filters metadata to create sets of volumes matching particular sets of "tags" specified by parameters in logisticpredict. Can also identify a set of volumes that are to be tested only -- never included in the training set.

metautils.py
------------
Simple utility for munging metadata, called by both logisticpredict and metafilter.

fiction_browser.py
------------------
In the data creation process, interactively selects fiction volumes in particular genres from HathiTrust metadata and adds them to metadata.

tag_chicago_data.py
-------------------
An analogous browsing script for Chicago metadata.

select_random_corpus.py
-----------------------
Selects random volumes and interactively allows user to approve them, plus provide enriched metadata.

logisticleave1out.py
--------------------
Old version of logisticpredict. Even on github, sometimes my code repos accumulate "old versions" that I'm afraid to throw out.

entropyfunctions.py
-------------------
This and several other scripts (poetic_entropy_parallelizer, others w/ 'entropy' in their names) are related to a project on conditional entropy that is only tangentially linked to the research on genre in this repo.
