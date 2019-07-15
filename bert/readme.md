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

In simplifying the HuggingFace code, Rajapakse took out the parts that permit running on multiple GPUs; regression is also only partly implemented in his version. I put those back. I didn't in the end use the regression feature; instead I edited the evaluation script so that it outputs raw logits as well as binary predictions. Running on multiple GPUs does significantly speed things up. I used a TeslaK40M for much of this, but for training on 512-word chunks with a batch size of 24 I needed K80s. The batch scripts I submit can specify particular hardware.


