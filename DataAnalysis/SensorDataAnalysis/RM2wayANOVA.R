setwd("")

# [Note added in 2021/07/07]
# For statistical analysis, I used anovakun (http://riseki.php.xdomain.jp/index.php?ANOVA%E5%90%9B).
source('anovakun_483.txt')

dat <- read.csv('./Dataset_PeakLatency.csv', header = F)
Results <- anovakun(dat, 'sAB', 3, 4, holm = T, mau = T, gg = T, peta = T, besci = T, tech = T)


#-- post hoc paired tests using Wilcoxon signed-rank test --#
#[1] main effect of category
NF <- (dat$V1+dat$V2+dat$V3+dat$V4) /4
FF <- (dat$V5+dat$V6+dat$V7+dat$V8) /4
Ho <- (dat$V9+dat$V10+dat$V11+dat$V12) /4

w1 <- wilcox.test(NF, FF, paired = T, exact = F, correct = F)
w2 <- wilcox.test(NF, Ho, paired = T, exact = F, correct = F)
w3 <- wilcox.test(FF, Ho, paired = T, exact = F, correct = F)

Pfdr <- p.adjust(c(w1$p.value, w2$p.value, w3$p.value), method = 'BH')

print(w1)
print(w2)
print(w3)
print(Pfdr)

rm(NF, FF, Ho, w1, w2, w3, Pfdr)


#[2] main effect of SF
BSF <- (dat$V1+dat$V5+dat$V9) /3
LSF <- (dat$V2+dat$V6+dat$V10) /3
HSF <- (dat$V3+dat$V7+dat$V11) /3
Equ <- (dat$V4+dat$V8+dat$V12) /3

w1 <- wilcox.test(BSF, LSF, paired = T, exact = F, correct = F)
w2 <- wilcox.test(BSF, HSF, paired = T, exact = F, correct = F)
w3 <- wilcox.test(BSF, Equ, paired = T, exact = F, correct = F)
w4 <- wilcox.test(LSF, HSF, paired = T, exact = F, correct = F)
w5 <- wilcox.test(LSF, Equ, paired = T, exact = F, correct = F)
w6 <- wilcox.test(HSF, Equ, paired = T, exact = F, correct = F)

Pfdr <- p.adjust(c(w1$p.value, w2$p.value, w3$p.value, w4$p.value, w5$p.value, w6$p.value), method = 'BH')

print(w1)
print(w2)
print(w3)
print(w4)
print(w5)
print(w6)
print(Pfdr)

rm(BSF, LSF, HSF, Equ, w1, w2, w3, w4, w5, w6, Pfdr)


#[3] Category X SF interaction
#[3-1] Category at each SF
NF <- dat$V2
FF <- dat$V6
Ho <- dat$V10

w1 <- wilcox.test(NF, FF, paired = T, exact = F, correct = F)
w2 <- wilcox.test(NF, Ho, paired = T, exact = F, correct = F)
w3 <- wilcox.test(FF, Ho, paired = T, exact = F, correct = F)

Pfdr <- p.adjust(c(w1$p.value, w2$p.value, w3$p.value), method = 'BH')

print(w1)
print(w2)
print(w3)
print(Pfdr)

rm(NF, FF, Ho, w1, w2, w3, Pfdr)


#[3-2] SF at each category
BSF <- dat$V9
LSF <- dat$V10
HSF <- dat$V11
Equ <- dat$V12

w1 <- wilcox.test(BSF, LSF, paired = T, exact = F, correct = F)
w2 <- wilcox.test(BSF, HSF, paired = T, exact = F, correct = F)
w3 <- wilcox.test(BSF, Equ, paired = T, exact = F, correct = F)
w4 <- wilcox.test(LSF, HSF, paired = T, exact = F, correct = F)
w5 <- wilcox.test(LSF, Equ, paired = T, exact = F, correct = F)
w6 <- wilcox.test(HSF, Equ, paired = T, exact = F, correct = F)

Pfdr <- p.adjust(c(w1$p.value, w2$p.value, w3$p.value, w4$p.value, w5$p.value, w6$p.value), method = 'BH')

print(w1)
print(w2)
print(w3)
print(w4)
print(w5)
print(w6)
print(Pfdr)

rm(BSF, LSF, HSF, Equ, w1, w2, w3, w4, w5, w6, Pfdr)
