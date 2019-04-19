gothic <- read.csv('~/Dropbox/fiction/variation/gothic_models.tsv', sep = '\t')
gothic <- data.frame(genre = rep('Gothic/horror', 100), accuracy = gothic$accuracy)
detect <-  read.csv('~/Dropbox/fiction/variation/detective_models.tsv', sep = '\t')
detect <- data.frame(genre = rep('detective', 100), accuracy = detect$accuracy)
scifi <- read.csv('~/Dropbox/fiction/variation/scifi_models.tsv', sep = '\t')
scifi <- data.frame(genre = rep('SF', 99), accuracy = scifi$accuracy)

variation <- rbind(gothic, detect, scifi)

pointestimates = data.frame(genre = as.factor(c('Gothic/horror', 'detective', 'SF')),
                            accuracy = c(.810, .934, .906))

library(ggplot2)
library(scales)

p <- ggplot(data = variation, aes(x = as.factor(genre), y = accuracy)) + theme_bw() +
  geom_boxplot(fill = 'gray85', width = 0.3, outlier.shape = NA) +
  geom_point(data = pointestimates, aes(x = genre, y = accuracy), shape = 9, size = 2) +
  scale_y_continuous('', labels = percent, breaks = seq(.80, .94, by = .02)) +
  theme(text = element_text(size = 20, family = "Baskerville"), panel.border = element_blank()) +
  theme(axis.line = element_line(color = 'black'),
        axis.text = element_text(color = 'black'),
        axis.title.x = element_blank())

tiff("/Users/tunder/Dropbox/active/criticalinquiry/boxplot.tiff", height = 5, width = 6, units = 'in', res=400)
plot(p)
dev.off()
plot(p)
 