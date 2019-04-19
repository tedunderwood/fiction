Sampling variation for predictive models
========================================

This is a simple repository that slightly improves my 2015 article "The Life Cycles of Genres" by more carefully measuring the effect of sampling variation on predictive accuracy.

In the original article, I make clear that I have different numbers of examples for each genre, and also make clear that there's a relationship between sample size and accuracy. But I don't go to great lengths to create apples-to-apples comparisons in every case, partly because absolute predictive accuracy is not the central point of the argument. (The article is more concerned with comparisons carried out by applying one model to a different set of data.)

However, for the sake of demonstrating how methods can be refined, I've circled back to this dataset to create strict apples-to-apples comparisons while also doing a better job of measuring variation.

The code here uses the data and metadata from the 2015 article. But the modeling code itself is more recent; it does a better job of optimizing grid search than my 2015 code did.

The script that matters for this process is **new_experiment.py**.  I produced the relevant results by running:

    > python3 new_experiment.py gothicvariations
    > python3 new_experiment.py sfvariations
    > python3 new_experiment.py detective variations

That created files in the ../results folder, but for simplicity I've moved them here: **detective_models.tsv, gothic_models.tsv** and **scifi_models.tsv.**
