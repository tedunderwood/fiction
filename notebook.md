Sketch of the experiment
========================

I'm gathering samples of fictional texts that are said to belong to particular genres by a particular authority at some particular time.

These particular statements might (or might not) all cohere in a larger genre system.

But we can test the coherence of the macroscopic genre. We can also see how that coherence changes over time, and test the relation of different authorities or subgenres to the superset.


Wed Aug 19
----------

State of play so far: I've gathered early mystery and detective fiction (to 1923) from bibliographies, as well as volumes to 1990 from the Chicago corpus. It models extremely well. Before 1923, accuracy is 92%. The stuff after '23 goes down to 85%, but I suspect that is largely because of errors in tagging (especially works that are mysteries but weren't tagged by librarians at all).

Right now I'm using default settings on the code I inherited from the reception project -- 3200 words, rgularization = .00007. Will fine-tune later, but at first glance they seem to work quite well for genre.

Theses in play at the moment: First, it's interesting that mysteries cohere at all over this span of time (150 years 1840-1990, or close thereto). Moretti's notion that genres are basically generational phenomena is going down. The first part of the timeline also predicts the second half of the timeline pretty well.

I need to figure out what the edges to this phenomenon are. What about sensation novels, Newgate novels? How well do they cohere?

Do all genres hold together as tightly as mysteries? Sci-fi really shouldn't be as _verbally_ cohesive, since the space of social reference is (sort of by definition) less defined. 

*afternoon*
However, it looks like in practice scifi is going to be almost as well-defined as mystery. We're getting 84% on a first pass, just using the LoC tags. An interesting detail is that accuracy and class separation seem stronger at the early end of the century (85.6% 1880-1950) than in recent decades (1950-1990 83.1%). Possibly scifi becoming a little more realist and quotidian in recent years? That could be a result.

However, we also need to find a genre that *doesn't* hold together, or I'm going to be in the paradoxical situation of having too much evidence.

*sanity check*
Random selection in the scifi sample produces 0.54% accuracy, which is a good sanity check. But there's an interesting taper even when you run *random* selection on this dataset. The two classes become slightly more like each other in the 20c -- even when the classes are randomly selected. That's something to keep an eye on.

Another good idea is to include the whole _Lock and Key Library_ as a test set of books identified as "detective fiction" that we expect not to match.

Thursday, August 20th
---------------------

Pushing hard here, I got a source (David Punter & Glennis Byron, _The Gothic_) that characterizes the Gothic as a long-running Otranto-to-Anne-Rice sort of phenomenon. Loaded about 74 examples from its "chronology" of the Gothic, and paired them with 74 random examples. We're getting, like, 72-74% accuracy, which is low enough to make me say that's a "genre" of rather a different sort than detective fiction.

It may not be accidental that the genres that seem relatively solid over long spans of time are genres that directly organized production and reception, whereas Punter & Byron's "gothic" is largely a retrospective phenomenon. Would also be interesting to discover whether there are particular zones of confusion. Right now it sort of looks like they're particularly astray in the middle of the 19c, where they invent continuity across the desert of realism.

Another open question is the relation between my different sets of 'random' volumes. Can they be rolled into a single classification? Probably. We could also tag some volumes as random-plus-something else, and those will be deselected when we're modeling that something else, according to the current behavior of metafilter.

Sunday, August 30th
-------------------
A lot of work done. I recreated the whole Chicago corpus with a better randomization strategy, so that it's possible for volumes to be in both the random dataset *and* the dataset for some particular genre. I also added pbgothic in the 20c, and anatscifi/femscifi in the 19c. Identified some missing volumes that need to be purchased.

But most importantly, reran logisticleave1out on the new dataset and identified 3 clear patterns that form the backbone of an article:

1. Gothic definitely weaker than scifi, which is weaker than detective/crime. 76 / 86 / 90% accuracy over comparable time spans. But all of this holds over much longer timespans than Moretti predicts. Need apples-to-apples comparison to a Moretti-Jockers genre category. This is largely what we might expect, and gives us confidence in the method, so that we have some chance of believing:

2. The Rise and Fall of Genre -- proposed title. There's a very marked pattern where genres differentiate in the 19c, reach peak differentiation around 1940-50, and then reconverge. It's clearest with detective/crime fiction. But also holds e.g. with crime vs. scifi. The convergence is actually clearer than the differentiation. Probably has to do with increasing quotidian realism in genre fiction, but that's a speculation at this point. Also, see "sanity check" above for an important source of doubt that needs to be explored.

3. The overarching thesis, uniting 1 & 2: "genre" is not a historically stable analytic category.
