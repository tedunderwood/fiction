gothic <- read.csv('~/Dropbox/fiction/variation/BoWGothic_models.tsv', sep = '\t')
gothic <- data.frame(genre = rep('Gothic/horror', 30), accuracy = gothic$validationacc)
detect <-  read.csv('~/Dropbox/fiction/variation/BoWMystery_models.tsv', sep = '\t')
detect <- data.frame(genre = rep('detective', 30), accuracy = detect$validationacc)
scifi <- read.csv('~/Dropbox/fiction/variation/BoWSF_models.tsv', sep = '\t')
scifi <- data.frame(genre = rep('SF', 30), accuracy = scifi$validationacc)

variation <- rbind(gothic, detect, scifi)

pointestimates = data.frame(genre = as.factor(c('Gothic/horror', 'detective', 'SF')),
                            accuracy = c(.77, .91, .883))

library(ggplot2)
library(scales)

p <- ggplot(data = variation, aes(x = as.factor(genre), y = accuracy)) + theme_bw() +
  geom_boxplot(fill = 'gray85', width = 0.3, outlier.shape = NA, coef = 3) +
  geom_point(data = pointestimates, aes(x = genre, y = accuracy), shape = 9, size = 2.5) +
  scale_y_continuous('', labels = percent, breaks = seq(.64, .96, by = .04)) +
  theme(text = element_text(size = 20, family = "Baskerville"), panel.border = element_blank()) +
  theme(axis.line = element_line(color = 'black'),
        axis.text = element_text(color = 'black'),
        axis.title.x = element_blank())

tiff("/Users/tunder/Dropbox/active/criticalinquiry/boxplot3.tiff", height = 5, width = 6, units = 'in', res=400)
plot(p)
dev.off()
plot(p)
 