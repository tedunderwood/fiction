Sampling variation for predictive models
========================================

This is a simple repository that slightly improves my 2015 article ["The Life Cycles of Genres"](http://culturalanalytics.org/2016/05/the-life-cycles-of-genres/) by more carefully measuring the effect of sampling variation on predictive accuracy. It documents figure 2 in a response article submitted to *Critical Inquiry* and tentatively titled "The Theoretical Divide Driving Debates about Computation."

In the original article, I make clear that I have different numbers of examples for each genre, and also make clear that there's a relationship between sample size and accuracy. But I don't go to great lengths to create apples-to-apples comparisons in every case, partly because absolute predictive accuracy is not the central point of the argument. (The article is more concerned with comparisons carried out by applying one model to a different set of data.)

However, for the sake of demonstrating how methods can be refined, I've circled back to this dataset to create strict apples-to-apples comparisons while also doing a better job of measuring variation.

The code here uses the data and metadata from the 2015 article. But the modeling code itself is more recent. It improves on my 2015 code by

1. Doing a better job of grid search.
2. Separating the training and test sets from a third set used only for validation. (This becomes necessary once I'm optimizing parameters using grid search.)
3. Running the process many times, so we can estimate variation.

The script that matters most for this process is **make_validation_splits.ipynb**.  It defines a split between a test-and-training set and a validation set. Then it calls versatiletrainer2.py, which optimizes a model of the test-and-training set through cross-validation. The model is written to disk, and tested on the validation set. We repeat the process thirty times for each genre.

That notebook creates results files included here as ```BoWMystery_models.tsv,``` etc, which in turn are used by boxplot.R to create the visualization used in the *Critical Inquiry* article. The point estimates also plotted in the visualization are taken from "The Life Cycles of Genres," 2015.
