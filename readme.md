fiction
=======

Code and data supporting the article "The Life Cycles of Genres" in _Cultural Analytics._ 

[![DOI](https://zenodo.org/badge/19804/tedunderwood/fiction.svg)](https://zenodo.org/badge/latestdoi/19804/tedunderwood/fiction)

The data model here assumes that genre designations are situated and perspectival. An observer in a particular place and time groups a particular set of works and calls them 'crime fiction.' We don't necessarily know that anyone else will agree; a different observer could group different works as 'detective fiction,' which might or not be the same thing. Nothing prevents some of these works from also being 'science fiction.' For that matter, some works can belong to no genre at all.

In short, every work can carry any number of genre tags, from zero upward. The compatibility of different definitions becomes an empirical question. Do different observers actually agree? Can a model trained on one observer's claims about detective fiction also predict the boundaries of 'crime fiction', as defined by someone else?

We use predictive modeling to test these questions. If you want to replicate the results here you'll need Python 3 and a copy of this repository. Running code/replicate.py will give you a range of modeling options keyed to particular sections of the article. The script will draw on metadata in meta/finalmeta.csv, wordcount files in the newdata directory, and the provided lexicon. Note that the selection of volumes in the negative contrast set can be stochastic, if more are available than needed to match the positive volumes. (For that matter, the positive set can at times be a random subset too.) So please don't expect replication to exactly match every figure down to the decimal point.

Because many of the books here are under copyright or otherwise encumbered with intellectual property agreements, I have to share wordcounts rather than original texts. If you want to consult texts in HathiTrust before 1922, it's usually possible to find them by pasting the Hathi volume id into a link of this form:

[http://babel.hathitrust.org/cgi/pt?id=uiuo.ark:/13960/t7wm20x0v](http://babel.hathitrust.org/cgi/pt?id=uiuo.ark:/13960/t7wm20x0v)

final
----
Assembling results and images to support publication as they approach readiness.

meta
----
Metadata for the project.

Right now the most complete set of metadata is in finalmeta.csv. 

newdata
----
The data used in the model: tables of word counts for each volume, as separate files. No, there is no olddata.

code
----
Code for the modeling process. The key modules for modeling are logisticpredict, metafilter, modelingprocess, and metautils. replicate.py is a script that allows readers to reproduce the particular settings I used for tests in the article. Not finalized yet, but getting close.

plot
----
(Mostly R) scripts for plotting in the sense of "dataviz." Has nothing to do with fictional plots.

lexicon
-------
The set of features that was used to produce the article; the top 10,000 words by document frequency in the whole corpus.
