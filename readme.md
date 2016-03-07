fiction
=======

Code and data for a project supporting the article, "The Life Cycles of Genres." 

The data model here assumes that genre designations are situated and perspectival. An observer in a particular place and time groups a particular set of works and calls them 'crime fiction,' or what have you. We don't necessarily know that anyone else will agree; a different observer could group different works as 'crime fiction,' or 'detective fiction.' Nothing prevents some of the works from also being 'science fiction.' For that matter, some works can belong to no genre at all.

In short, every work can carry any number of genre tags, from zero upward. The compatibility of different definitions becomes an empirical question. Do different observers actually agree about detective fiction? Can a model trained on one observer's claims about detective fiction also predict crime fiction?

We use predictive modeling to test these questions.

final
----
Assembling results and images to support publication as they approach readiness.

meta
----
Metadata for the project.

Right now the most complete set of metadata is in finalmeta.csv. 

data
----
The data used in the model: tables of word counts for each volume, as separate files.

code
----
(Mostly Python) code. The key modules for modeling are logisticpredict, metafilter, modelingprocess, and metautils. replicate.py is (going to be) the script that allows readers to reproduce the particular settings I used for tests in the article. Not finalized yet.

plot
----
(Mostly R) scripts for visualization.

workflow
--------

The dataset-creation workflow is a mess because I'm assembling volumes from multiple sources and doing a lot of manual tagging. But it starts by randomly selecting volumes from existing corpora, guided by LoC tags, using scripts like get_chicago_data and find_genre_fiction. I add volumes one by one from bibliographies, using fiction_browser and tag_chicago_data. 

This leaves me with two separate metadata files (hathigenremeta and chicagometa). To actually get the hathi data I run list_of_missing_vols and then rsync the resulting volume list up to Taub.

At that point, two commands in code/extract:

python3 extract.py -v -g fic -idfile filestoget2015-08-30.txt -o /projects/ichass/usesofscale/sampletexts/ -rh

to get pre1900 and

python3 extract.py -v -g fic -idfile filestoget2015-08-30.txt -o /projects/ichass/usesofscale/sampletexts/ -rh -index 20cpredictions.index -root /projects/ichass/usesofscale/20c/

to get post1900

Equivalent chicago process starts with gather_chicago, and proceeds through 

python NormalizeOneFolder.py /Volumes/TARDIS/US_Novel_Corpus/selected/ /Volumes/TARDIS/US_Novel_Corpus/selectcounts/
