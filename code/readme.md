fiction/code
============

replicate.py
------------
The central script that you call to reproduce tests from the article. If you run it interactively, it will query you and ask you which part of the project you want to reproduce. You can also specify other settings to compare other sets of genres.

logisticpredict.py
------------------
The main modeling script, called by replicate.py with specific parameters. Does leave-one-out crossvalidation by author.

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
