library(scales)
library(ggplot2)

stewacc <- scan('/Users/tunder/Dropbox/fiction/code/stewaccuracies.csv')
randomacc <- scan('/Users/tunder/Dropbox/fiction/code/finalaccuracies.csv')
detectiveacc <- scan('/Users/tunder/Dropbox/fiction/code/detaccuracies.csv')
allacc <- data.frame(acc = c(randomacc, stewacc, detectiveacc),
                     type = c(rep('\nrandom\n', 40), rep('\ngenre\n"stew"\n', 40), rep('\ndetective\n', 40)))

p <- ggplot(allacc, aes(x = acc, colour = type, fill = type)) + geom_histogram(binwidth = 0.01) +
  scale_colour_manual(values = c('black', 'black', 'black')) + scale_fill_manual(values = c('coral', 'dodgerblue', 'gray50')) +
  theme(text = element_text(size = 16)) +
  theme(legend.title=element_blank()) +
  scale_y_continuous('count', breaks = c(2,4,6,8,10,12,14,16)) +
  scale_x_continuous('\naccuracy of 40 models for each group,\nwith 140 positive and negative instances in each model', limits = c(0.3,1), labels = percent, breaks = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1))
plot(p)