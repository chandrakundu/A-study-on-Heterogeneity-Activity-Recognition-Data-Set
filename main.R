library("textclean")
library(dplyr)
library(gutenbergr)
library(tidytext)
library(tm)

#load the function words
func_word <- read.csv("func_word.csv", stringsAsFactors=FALSE) %>% select(word)
# func_word <- vecToDF(function.words, "word")

# Directory of the corpus of authors
# setwd("C:/Users/virus/Google Drive/Current/Thesis/AuthorshipAttribution")
dirBaum <- "Oz/baum/"
dirRoyal <- "Oz/royal2/"
dirThompson <- "Oz/thompson/"
dirJacksnow <- "Oz/jacksnow/"
dirHGwells <- "Oz/hg/"


# Gutenberg ids for download
# https://sites.google.com/site/freechildrensbooks/home/oz-books
#oz_baum <- c(43936, 54, 486, 22566, 485)
# oz_baum <- c(41666, 32094, 52176, 52263, 25581, 24459, 30852, 419, 961 )
 royal_oz <- 30537
 # oz_thompson <- c(53765, 58765, 55851, 56073, 56079, 56085, 55806)
# oz_thompson <- c(53765, 55851)
oz_jacksnow <- c(56683, 56555)
# oz_others <- c(10127, 10419)
# hgwells <- c(58877, 45368, 27365, 6424 )


# download baum books
# downloadGutenberg(oz_baum, dirBaum)
# downloadGutenberg(oz_thompson, dirThomson)
# downloadGutenberg(hgwells, dirHGwells)
downloadGutenberg(royal_oz, dirRoyal)


# read data
baumBooks <- readData(dirBaum)
royalOz <- readData(dirRoyal)
thomsonBooks <- readData(dirThomson)
jacksnowBooks <- readData(dirJacksnow)


#find prob distribution
baumDist <- findDistribution(baumBooks, func_word)
royalDist <- findDistribution(royalOz, func_word)
thomsonDist <- findDistribution(thomsonBooks, func_word)
jacksnowDist <- findDistribution(jacksnowBooks, func_word)


# Create plot
m = 100
n = 10000
alpha = 0.05


par(mfrow = c(2,2))
createPlot(baumDist, royalDist, m, n, label = "Baum Vs Royal when n = 10000")
createPlot(thomsonDist, royalDist, m, n, label = "Thomson Vs Royal when n = 10000")
createPlot(jacksnowDist, royalDist, m, n, label = "jacksnow Vs Royal when n = 10000")
createPlot(baumDist, thomsonDist, m, n, label = "jacksnow Vs Royal when n = 10000")



# baum vs baum
baum1 <- readData("Oz/baum1")
baum2 <- readData("Oz/baum2")

baum1dist <- findDistribution(baum1, func_word)
baum2dist <- findDistribution(baum2, func_word)

createPlot(baum1dist, baum2dist, m = 50, n, label = "Baum Vs Royal when n = 10000")
createPlot(baum1dist, baum2dist, m = 50, n, label = "Baum Vs baum when n = 10000")

# thomson vs thomson 
thomson1 <- readData("Oz/thomson1")
thomson2 <- readData("Oz/thomson2")

thomson1dist <- findDistribution(thomson1, func_word)
thomson2dist <- findDistribution(thomson2, func_word)

createPlot(thomson1dist, thomson2dist, m = 50, n=1000, label = "Baum Vs Royal when n = 1000")

# m = 50 and n = 1000
createPlot(baumDist, royalDist, m = 50, n=1000, label = "Baum Vs royal when n = 1000")
createPlot(thomsonDist, royalDist, m = 50, n=1000, label = "thomson Vs royal when n = 1000")


# thomson gutenberg
thomsonG <- readData(dirThomson)
thomsonGdist <- findDistribution(thomsonG, func_word)
thomson <- readData("Oz/others/")
thomsondist <- findDistribution(thomson, func_word)

createPlot(thomsondist, royalDist, m = 50, n=1000, label = "thomson Vs royal when n = 1000")




