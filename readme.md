fiction
=======

Code and data for a project exploring the history of fictional genres. 

The strategy is not to rely on any single definition of genre, but to gather a bunch of _different_ sources of social testimony about overlapping genre categories. "Librarians have tagged these books as 'science fiction,' but here's a more inclusive bibliography of 'speculative fiction,' so let's record that too -- and so on."

We'll then use predictive modeling to map relations of similarity and dissimilarity between these different definitions of genre, and, ultimately, to trace macroscopic histories of generic differentiation or convergence.

final
----
Assembling results and images to support publication as they approach readiness.

meta
----
Metadata for the project.

Right now the most complete set of metadata is in finalmeta.csv. 

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
