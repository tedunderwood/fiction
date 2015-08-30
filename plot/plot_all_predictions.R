library(scales)
library(ggplot2)
library(dplyr)
l <- read.csv('~/Dropbox/fiction/results/crimepredictions.csv')
reviewd <- as.factor(l$realclass)
levels(reviewd) <- c('not', 'rev')
l$reviewed <- reviewd

model = lm(data = filter(l, realclass == 1), formula = logistic ~ dateused)
mintercept = coef(model)[1]
mslope = coef(model)[2]
model = lm(data = filter(l, realclass == 0), formula = logistic ~ dateused)
rintercept = coef(model)[1]
rslope = coef(model)[2]

# Now let's do the main model
model = lm(data = l, formula = logistic ~ dateused)
intercept = coef(model)[1]
slope = coef(model)[2]

line = intercept + (slope * l$dateused)
true = sum(l$logistic > line & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic <= line & l$reviewed == 'not', na.rm = TRUE)
false = sum(l$logistic < line & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic >= line & l$reviewed == 'not', na.rm = TRUE)
print(true / (true + false))
levels(l$reviewed) = c('detective', 'SF')
p <- ggplot(l, aes(x = dateused, y = logistic, color = reviewed, shape = reviewed)) + 
  geom_point() + geom_abline(intercept = rintercept, slope = rslope) + 
  geom_abline(intercept = mintercept, slope = mslope, colour = 'red') + scale_shape(name="actually") + 
  scale_color_manual(name = "actually", values = c('gray60', 'red2')) + 
  theme(text = element_text(size = 16)) + 
  scale_y_continuous('Predicted prob. of beng science fiction\n', labels = percent, breaks = c(0.25,0.5,0.75)) + 
  scale_x_continuous("")
plot(p)

periodaccuracy <- function(start, end) {
  true <- 0
  true <- true + sum(l$logistic > line & l$realclass == 1 & l$dateused >= start & l$dateused <= end, na.rm = TRUE)
  true <- true + sum(l$logistic <= line & l$realclass == 0 & l$dateused >= start & l$dateused <= end, na.rm = TRUE)
  false <- 0
  false <- false + sum(l$logistic < line & l$realclass == 1 & l$dateused >= start & l$dateused <= end, na.rm = TRUE)
  false <- false +  sum(l$logistic >= line & l$realclass == 0 & l$dateused >= start & l$dateused <= end, na.rm = TRUE)
  print( true / (false + true))
}
l$error = abs(as.numeric(l$realclass) - l$logistic)
arrange(select(l, title, author, dateused, error), error)