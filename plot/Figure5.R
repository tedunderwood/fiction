# accuracycurves.R
library(msir)
det <- read.csv('/Users/tunder/Dropbox/fiction/final/collateddetaccuracies.tsv', sep = '\t')
stew <- read.csv('/Users/tunder/Dropbox/fiction/final/collatedstewaccuracies.tsv', sep = '\t')

# det$genre <- as.factor(rep('detective', dim(det)[1]))
# stew$genre <- as.factor(rep('stew', dim(stew)[1]))

## a different approach

## a different approach

makeribbon <- function(frame, label) {
  framelen = dim(frame)[1]
  f <- loess(rawaccuracy ~ avgsize, frame, span = 0.85)
  predictions = predict(f)
  errors <- abs(f$y - predictions)
  up = c()
  low = c()
  
  lastradius = 100
  xseq = frame$avgsize
  for (i in 1:framelen) {
    start = i - 7
    if (start < 1) start = 1
    end = i + 7
    if (end > framelen) end = framelen
    
    meanerror = mean(errors[start:end])
    sderror = sd(errors[start:end])
    intervalradius = meanerror + (sderror * 0.82)
    if (xseq[i] > 90 & intervalradius > lastradius) {
      intervalradius = lastradius
    }
    lastradius = intervalradius
    
    # This calculates a 90% prediction band; in other words, 90% of
    # data points should lie within the shaded area. It's 0.82
    # instead of 1.64 because I'm doing this one-tailed; i.e., for
    # one half of the band.
    
    # The business about lastradius is a kludge because it's computationally
    # expensive to run a lot of large models.
    # I know the variance of prediction accuracy doesn't actually increase,
    # so I just don't let it increase.
    
    up <- c(up, predictions[i] + intervalradius)
    low <- c(low, predictions[i] - intervalradius)
  }
  up = predict(loess(y ~ x, data.frame(x = xseq, y = up)))
  low = predict(loess(y ~ x, data.frame(x = xseq, y = low)))
  df <- data.frame(x = frame$avgsize/2, upperbound = up, lowerbound = low, line = predictions, genre = as.factor(rep(label, framelen)))
  return(df)
}

df <- makeribbon(det, '\ndetective\n')

samplesizes = c(12.1, 26.4, 40.3, 18.4, 18.5, 105.5)
accuracies = c(.5, .75, .810, .65, .850,.87675)
points = data.frame(x = samplesizes/2, y = accuracies)

p <- ggplot(df, aes(x = x)) + geom_line(aes(y = line, color = genre)) + 
  geom_ribbon(data=subset(df, genre == '\ndetective\n'), aes(ymin=lowerbound, ymax=upperbound), alpha=0.2) +
  geom_point(data = points, aes(x = x, y = y)) + theme(text = element_text(size = 16)) +
  scale_x_continuous("number of examples of the genre in the training set") +
  scale_y_continuous('accuracy\n', labels = percent) + scale_color_manual(values = c('red2', 'darkcyan'))

plot(p)
