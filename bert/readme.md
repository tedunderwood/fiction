BERT scripts
============

Scripts for testing BERT and comparing it to bag-of-words models. Related to a blog post:

The BERT code
-------------

If you have PyTorch set up on your machine, it should be possible to reuse much of this code. It ran under Python 3.7.3 and PyTorch 0.4.1.post2.

The code for implementing BERT on PyTorch is borrowed ultimately [from HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT), but proximally from ["A Simple Guide on Using BERT for Binary Text Classification," by Thilina Rajapakse](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04). In particular, I borrow Rajapakse's method of parallelizing the tokenization process, which valuably speeds up that step.

I then edited the code myself to frame it differently. Rajapakse runs BERT inside a notebook, but my hardware doesn't allow that (since I'm working through a batch queue).

So I divided the workflow into four stages, and constructed separate python scripts to

1. **Create** data for BERT. This involves balancing classes, distributing them evenly across the timeline, making sure no authors are present in both training and test sets, and finally dividing the texts into BERT-sized chunks. I make an effort to start the chunks at grammatical boundaries. This part can be done in a notebook: **create_bert_sample.ipynb.**

2. **Convert** training data into BERT format, using the BERT tokenizer in PyTorch. The pbs script **convert512.pbs** invokes **converter512.py.**

3. **Train** models on the converted data. The script **train512.pbs** invokes **trainbert512.py**.

4. **Eval**uate the models, using a separate validation set. (This script also tokenizes the validation data, combining steps that are separated in 2 and 3; while this may seem illogical, it can make sense when you're limited by a 4-hour clock in a batch queue. Step 2 is usually the time-consuming bottleneck.) Evaluation is done by **eval512.pbs,** which invokes **evalbert512.py**.

Because I needed to try lots of different settings on three different genres, I needed a way to keep different trials separate. Changing filenames is not a great solution, because BERT expects data and models to have predictable names (e.g., the training data is called "train.tsv" and the test set is "dev.tsv.")

The solution I adopted was to create separate folders for each trial under **data/, outputs/,** and **reports/**.  So, for instance, when I was running 256-word chunks of Gothic fiction, the data came from **data/goth256,** the model went in **outputs/goth256,** and the reports/results got written to **reports/goth256.**

I passed the trial code (or TASK_NAME) to the Python script as a command-line argument, which meant that I needed to edit the PBS (batch) script in each case.

You will also notice that the code described above has the number "512" embedded in filenames. As I tinkered, I discovered that BERT does a lot better with genre if it can see longer chunks. I could have passed this parameter also as a command-line argument, but instead I spun off a separate series of scripts to model the longer 512-word chunks of text.

In simplifying the HuggingFace code, Rajapakse took out the parts that permit running on multiple GPUs; regression is also only partly implemented in his version. I put those back. I didn't in the end use the regression feature much; instead I edited the evaluation script so that it outputs raw logits as well as binary predictions. Running on multiple GPUs does significantly speed things up. I used a TeslaK40M for much of this, but for training on 512-word chunks with a batch size of 24, I had to specify K80s. The batch scripts I submit can specify particular hardware.

The optimal settings for genre seemed to be 512-word chunks, with a batch size of 24, and 2 epochs of training. I kept learning rate, warmup, etc. set to the defaults provided by HuggingFace. More tuning is probably possible, although at some point one would need to construct a separate validation set to confirm that we're not overfitting the parameters.

Interpreting BERT results
-------------------------

The code I borrowed does print a report that lists numbers of true positives, false positives, and so on. But since I was working with long documents, I wasn't really concerned with BERT's raw predictions about individual text chunks. Instead I needed to know how good the predictions would be if we aggregated them at the volume level.

To figure this out I used **interpret_bert.ipynb**. This notebook pairs BERT's predictions with a metadata file that got spun off back in step 1 above, called, for instance, **bertmeta/dev_rows_{TASK_NAME}.tsv**. This metadata file lists the index of each chunk but also the **docid** (usually, volume-level ID) associated with a larger document.

The **interpret_bert** notebook can then group predictions by volume and evaluate accuracy at the volume level. I tried doing this by averaging logits, as well as binary voting. I think in most cases binary voting is preferable; I'm not sure whether the logits are scaled in a way that produces a reliable mean.


Comparisons to BoW models
-------------------------

The really rigorous modeling of genre using bag-of-words methods took place [in a separate directory.](https://github.com/tedunderwood/fiction/tree/master/variation) There I have [a notebook](https://github.com/tedunderwood/fiction/blob/master/variation/make_validation_splits.ipynb) that does a better job of explaining how I construct balanced training, test, and validation sets, and repeat the process to model random variation. This is the process that produced the boxplots.

But I also needed to produce a lot of more casual models to see how bag-of-words methods suffer when forced to use a smaller window. I did that in this directory, inside the notebook **logistic_regression_touchstones.ipynb.**




