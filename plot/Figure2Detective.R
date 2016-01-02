library(scales)
library(ggplot2)
library(dplyr)
l <- read.csv('~/Dropbox/fiction/final/just249detective2015-12-30.csv')
reviewd <- as.factor(l$realclass)
levels(reviewd) <- c('not', 'rev')
l$reviewed <- reviewd

true = sum(l$logistic > 0.5 & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic <= 0.5 & l$reviewed == 'not', na.rm = TRUE)
false = sum(l$logistic < 0.5 & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic >= 0.5 & l$reviewed == 'not', na.rm = TRUE)
print(true / (true + false))
levels(l$reviewed) = c('random\n\n', 'detective\nfiction:\nall sources')
p <- ggplot(l, aes(x = dateused, y = logistic, color = reviewed, shape = reviewed)) + 
  geom_point() + scale_shape(name="actually\n") + 
  scale_color_manual(name = "actually\n", values = c('gray60', 'red2')) + 
  theme(text = element_text(size = 16)) + 
  scale_y_continuous('Predicted probability of being detective fiction\n', labels = percent, breaks = c(0.25,0.5,0.75)) + 
  scale_x_continuous("", breaks = c(1850,1900,1950,1989)) + ggtitle('\n\n')
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