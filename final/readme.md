final data
==========
This is where I'm storing the final output of various models, which I've used to create the images in the article itself. Each model consists of a .csv file, which stores information about the volumes included, and a .coefs.csv file, which reports the coefficients in the model itself. The third column in coefs.csv should probably be ignored; it's a coefficient normalized for frequency, but in a list of 10,000 words that's unlikely to be helpful.

Under each model, I've explained how it was produced. Often this points back to a particular option in code/replicate.py. You should be able to produce a closely analogous model by running that option. (It may not be exactly identical, because there's a stochastic aspect to the selection of a random contrast set.)

ghastlystew2015-11-27
---------------------
was produced by combining all the genres included in the variable allstewgenres in the function "ghastly_stew" of replicate.py

IndianaDetective2016-04-09
--------------------------
was produced by option #1 in code/replicate.py

LOCDetective2016-04-10
----------------------
was produced by option #2 in code/replicate.py

AllDetective2016-04-09
----------------------
was produced by option #3 in code/replicate.py

detectivejustpost19302016-04-10
-------------------------------
was produced by option #5 in code/replicate.py
It's a benchmark for predicting detective fiction post-1930.

detectivepredictpost19302016-04-10
----------------------------------
was also produced by option #5 in code/replicate.py
It extrapolates a model of detective fiction pre-1930 to predict after 1930.

AllGothic2016-04-10
-------------------
was produced by option #14 in code/replicate.py

AllSF2016-04-10
---------------
was produced by option #15 in code/replicate.py