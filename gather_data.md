gather_data.md

The dataset-creation workflow is complex because I'm assembling volumes from multiple sources and doing a lot of manual tagging. But it starts by randomly selecting volumes from existing corpora, guided by LoC tags, using scripts like get_chicago_data and find_genre_fiction. I add volumes one by one from bibliographies, using fiction_browser and tag_chicago_data.

This leaves me with two separate metadata files (hathigenremeta and chicagometa). To actually get the hathi data I run list_of_missing_vols and then rsync the resulting volume list up to Taub.

At that point, two commands in code/extract:

python3 extract.py -v -g fic -idfile filestoget2015-08-30.txt -o /projects/ichass/usesofscale/sampletexts/ -rh

to get pre1900 and

python3 extract.py -v -g fic -idfile filestoget2015-08-30.txt -o /projects/ichass/usesofscale/sampletexts/ -rh -index 20cpredictions.index -root /projects/ichass/usesofscale/20c/

to get post1900

Equivalent chicago process starts with gather_chicago, and proceeds through

python NormalizeOneFolder.py /Volumes/TARDIS/US_Novel_Corpus/selected/ /Volumes/TARDIS/US_Novel_Corpus/selectcounts/