setwd("~/Downloads")

data = read.csv("Train.csv")
test = read.csv("Test_Data.csv")

#spiltting and initialization
library(caret)
library(caTools)
split = sample.split(data$label, SplitRatio = 0.7)
train = subset(data, split==TRUE)
cv = subset(data, split==FALSE)
tr.res = train$label
cv.res = cv$label
train$label = NULL
cv$label = NULL

#PCA
train = train[,apply(train, 2, var, na.rm=TRUE) != 0]
prin_comp = prcomp(train, scale. = T)
std_dev = prin_comp$sdev
pr_var = std_dev^2
prop_varex <- pr_var/sum(pr_var)
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")
train.data = data.frame(label = tr.res, prin_comp$x)
train.data = train.data[,1:200]
cv.data = predict(prin_comp, newdata = cv)
cv.data = as.data.frame(cv.data)
cv.data = cv.data[,1:200]


#Randomforest (option1 - cv(94.36%))
library(randomForest)
rf.model = randomForest(as.factor(label)~., data=train.data)
rf.pred = predict(rf.model, cv.data)
confusionMatrix(cv.res,rf.pred)




#---------------------------
#LDA (option2 - cv(86.61%))
library(MASS)
lda.model = lda(label~., data=train.data)
lda.prediction = MASS:::predict.lda(lda.model, cv.data)
lda.pred = lda.prediction$class
confusionMatrix(cv.res,lda.pred)

