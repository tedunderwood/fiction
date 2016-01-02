library(scales)
library(ggplot2)
library(dplyr)
l <- read.csv('~/Dropbox/fiction/final/newcrime2016-01-01.csv')
reviewd <- as.factor(l$realclass)
levels(reviewd) <- c('not', 'rev', 'newgate', 'sensation', 'crime')
l$reviewed <- reviewd

levels(l$reviewed) = c('\nrandom\n', '\ndetective\n', '\nnewgate\n', '\nsensation\n', '\ncrime\n')
p <- ggplot(l, aes(x = dateused, y = logistic, color = reviewed, shape = reviewed, size = reviewed)) + 
  geom_point() + scale_shape_manual(name="actually\n", values = c(16,17,15,18,7)) + 
  scale_color_manual(name = "actually\n", values = c('gray70', 'red2', 'chartreuse3', 'darkcyan', 'blue')) + 
  theme(text = element_text(size = 16)) + 
  scale_size_manual(name = "actually\n", values = c(2,2,3,3,3)) +
  scale_y_continuous('Predicted probability of beng detective fiction\n', labels = percent, breaks = c(0.25,0.5,0.75)) + 
  scale_x_continuous("", breaks = c(1850,1900,1950,1989)) 
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