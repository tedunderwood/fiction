# accuracycurves.R

det <- read.csv('/Users/tunder/Dropbox/fiction/final/collateddetaccuracies.tsv', sep = '\t')
stew <- read.csv('/Users/tunder/Dropbox/fiction/final/collatedstewaccuracies.tsv', sep = '\t')

# det$genre <- as.factor(rep('detective', dim(det)[1]))
# stew$genre <- as.factor(rep('stew', dim(stew)[1]))

## a different approach

makeribbon <- function(frame, label) {
  framelen = dim(frame)[1]
  f <- loess(rawaccuracy ~ avgsize, frame)
  predictions = predict(f)
  errors <- abs(f$y - predictions)
  up = c()
  low = c()
  lasterror = 0
  for (i in 1:framelen) {
    start = i - 7
    if (start < 1) start = 1
    end = i + 7
    if (end > framelen) end = framelen
    meanerror = mean(errors[start:end])
    # meanerror = sd(predictions[start:end]) / sqrt(end-start)
    if (lasterror > 0) meanerror = (meanerror + lasterror) / 2
    lasterror = meanerror
    up <- c(up, predictions[i] + meanerror)
    low <- c(low, predictions[i] - meanerror)
  }
  df <- data.frame(x = frame$avgsize/2, upperbound = up, lowerbound = low, line = predictions, genre = as.factor(rep(label, framelen)))
  return(df)
}

predictdet <- makeribbon(det, '\ndetective\n')
predictstew <- makeribbon(stew, '\nmixture\n')

df <- rbind(predictdet, predictstew)

samplesizes = c(38.6, 100, 186.5, 326.3)
accuracies = c(.833, .788, .768, .767)
points = data.frame(x = samplesizes/2, y = accuracies)

p <- ggplot(df, aes(x = x)) + geom_line(aes(y = line, color = genre)) + 
  geom_ribbon(data=subset(df, genre == '\ndetective\n'), aes(ymin=lowerbound, ymax=upperbound), alpha=0.2) +
  geom_ribbon(data=subset(df, genre == '\nmixture\n'), aes(ymin=lowerbound, ymax=upperbound), alpha=0.2) +
  geom_point(data = points, aes(x = x, y = y)) + theme(text = element_text(size = 16)) +
  scale_x_continuous("number of examples of the genre in the training set") +
  scale_y_continuous('accuracy\n', labels = percent)

plot(p)
