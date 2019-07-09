BERT scripts
============

Scripts for running BERT on the data underlying "Life Cycles of Genres." Much of the code is adapted from ["A Simple Guide on Using BERT for Binary Text Classification," by Thilina Rajapakse](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04).

This is not going to be smoothly push-button reproducible for people with different hardware; some editing of the workflow will be required. But here's what I did:

I run create_bert_sample.ipynb to generate multiple data and metadata files in local folders ```bertmeta``` and ```bertdata.```

Then I copy the train and dev data files one by one to the campus cluster. This has to be done one by one because my BERT scripts expect data files to have the generic names "train.tsv" and "dev.tsv." So I can't load multiple data files at once; in any case the data is large enough to make rsync slow.

So for instance I say

    rsync bertdata/train_SF0.tsv tunder@cc-login.campuscluster.illinois.edu:/projects/ischoolichass/ichass/usesofscale/code/lifecycle/bert/data/train.tsv

Then I run convert.pbs to convert the .tsv file to a pickled ```train_features``` file. This PBS script invokes converter.py and sends it a command-line argument, the TASK_NAME, which will determine, e.g. which subfolder of the outputs dir and reports dir get used.

