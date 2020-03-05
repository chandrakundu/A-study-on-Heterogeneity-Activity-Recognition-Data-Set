library("textclean")
library(dplyr)
library(gutenbergr)
library(tidytext)
library(tm)
library(ggplot2)
library(gridExtra)

# function word list
func.word <- read.csv("func_word.csv", stringsAsFactors=FALSE) %>% select(word)

# L. Frank Baum Oz 
baum.oz.dist <- findDistribution(readData("Oz/baum-oz/"))

# H. G. Wells
hgwells.1.dist <- findDistribution(readData("Oz/hgwells1/"))
hgwells.2.dist <- findDistribution(readData("Oz/hgwells2/"))
hgwells.dist <- findDistribution(readData("Oz/hgwells/"))

# L. Frank Baum other random books
baum.other.dist <- findDistribution(readData("Oz/baum-other/"))

# Ruth Plumly Thompson Oz Distribution
thompson.oz.dist <- findDistribution(readData("Oz/thompson-oz/"))

# The Royal Book of Oz Distribution
royal.book.oz.dist <- findDistribution(readData("Oz/royal-book-oz/"))

# Jack Snow Oz
jacksnow.oz.dist <- findDistribution(readData("Oz/jacksnow/"))



# Create condfidence interval for their mean difference and plot 
m = 100
n = 1000
alpha = 0.05


# L. Frank Baum Oz vs Others
baumoz.baumother <- confintPlot(baum.oz.dist, baum.other.dist, m, n, alpha, label = "L. Frank Baum: Oz series vs other books")

# H. G. Wells 1 vs H. G. Wells 2
hgwells1.hgwells2 <- confintPlot(hgwells.1.dist, hgwells.2.dist, m, n, alpha, label = "H. G. Wells 1 vs H. G. Wells 2")

# L. Frank Baum Oz vs H. G. Wells
baumoz.hgwells <- confintPlot(baum.oz.dist, hgwells.dist, m, n, alpha, label = "L. Frank Baum Oz vs H. G. Wells")

# L. Frank Baum Oz vs Ruth Plumly Thompson Oz
baumoz.thompsonoz <- confintPlot(baum.oz.dist, thompson.oz.dist, m, n, alpha, label = "L. Frank Baum Oz vs Ruth Plumly Thompson Oz")

# L. Frank Baum Oz vs Royal Book of Oz
baum.royal <- confintPlot(baum.oz.dist, royal.book.oz.dist, m, n, alpha, label = "L. Frank Baum Oz vs Royal Book of Oz")

# Ruth Plumly Thompson Oz vs Royal Book of Oz
thompson.royal <- confintPlot(thompson.oz.dist, royal.book.oz.dist, m, n, alpha, label = "Ruth Plumly Thompson Oz vs Royal Book of Oz")


# Jack Snow vs all
#jacksnow.hgwells <- confintPlot(jacksnow.oz.dist, hgwells.dist, m, n, alpha, label = "Jack Snow vs H. G. Wells")
baum.jack <- confintPlot(baum.oz.dist, jacksnow.oz.dist, m, n, alpha, label = "L. Frank Baum Oz vs Jack Snow Oz")
thompson.jacksnow <- confintPlot(thompson.oz.dist, jacksnow.oz.dist, m, n, alpha, label = "Ruth Plumly Thompson Oz vs Jack Snow Oz")
jacksnow.royal <- confintPlot(jacksnow.oz.dist, royal.book.oz.dist, m, n, alpha, label = "Jack Snow Oz vs Royal Book of Oz")


grid.arrange(baumoz.baumother, hgwells1.hgwells2)
grid.arrange(baumoz.hgwells, baumoz.thompsonoz, baum.jack, thompson.jacksnow, nrow=2)
grid.arrange(baum.royal, thompson.royal, jacksnow.royal, nrow=2)

