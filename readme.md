fiction
=======

Code and data for a project supporting the article, "The Life Cycles of Genres." 

The data model here assumes that genre designations are situated and perspectival. An observer in a particular place and time groups a particular set of works and calls them 'crime fiction,' or what have you. We don't necessarily know that anyone else will agree; a different observer could group different works as 'crime fiction,' or 'detective fiction.' Nothing prevents some of the works from also being 'science fiction.' For that matter, some works can belong to no genre at all.

In short, every work can carry any number of genre tags, from zero upward. The compatibility of different definitions becomes an empirical question. Do different observers actually agree about detective fiction? Can a model trained on one observer's claims about detective fiction also predict crime fiction?

We use predictive modeling to test these questions. If you want to replicate the results here you'll need Python 3 and a copy of this repository. Running code/replicate.py will give you a range of modeling options keyed to particular aspects of the article. It will draw on metadata in meta/finalmeta.csv, and wordcount files in the newdata directory.

Because many of the books here are under copyright or otherwise encumbered with intellectual property agreements, I cannot easily share the original texts.

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

