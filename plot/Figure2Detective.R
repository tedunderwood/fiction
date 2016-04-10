library(scales)
library(ggplot2)
library(dplyr)
l <- read.csv('~/Dropbox/fiction/results/AllDetective2016-04-09.csv')
reviewd <- as.factor(l$realclass)
levels(reviewd) <- c('not', 'rev')
l$reviewed <- reviewd

theshapes = rep('1', length(l$reviewed))
theshapes[l$reviewed == 'not'] <- 2
marked <- c('inu.30000007703436', 'uc2.ark+=13960=t45q4tp5h', '10889', 'nyp.33433074970710',
            'inu.30000007708815', '11098')
theshapes[which(l$volid %in% marked)] <- 3
l$shape <- as.factor(theshapes)

true = sum(l$logistic > 0.5 & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic <= 0.5 & l$reviewed == 'not', na.rm = TRUE)
false = sum(l$logistic < 0.5 & l$reviewed == 'rev', na.rm = TRUE) + sum(l$logistic >= 0.5 & l$reviewed == 'not', na.rm = TRUE)
print(true / (true + false))
levels(l$reviewed) = c('random\n\n', 'detective\nfiction:\nall sources')
p <- ggplot(l, aes(x = dateused, y = logistic, color = reviewed, shape = shape)) + 
  geom_point() + 
  scale_color_manual(name = "actually\n", values = c('gray60', 'red2')) + 
  scale_shape_manual(guide = FALSE, values = c(17, 16, 2)) +
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