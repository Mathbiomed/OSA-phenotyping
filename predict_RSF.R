library(randomForestSRC)
library(pec)

load(file='RSF_model.rda')

testset = read.csv('test_input.csv', row.names='index')
#testset = read.csv('random_testset.csv', row.names='index')

mean = read.csv('population_mean.csv', header=FALSE)
std = read.csv('population_std.csv', header=FALSE)

mean = t(mean)
std = t(std)

testset = (testset-mean)/std
names(testset) = names(rf.grow$xvar)

prob_5 = predictSurvProb(rf.grow, newdata = testset, times=5)
prob_10 = predictSurvProb(rf.grow, newdata = testset, times=10)
prob_15 = predictSurvProb(rf.grow, newdata = testset, times=15)

print(paste0("Predicted 5-year comorbidity-free survival probability: ", prob_5))
print(paste0("Predicted 10-year comorbidity-free survival probability: ", prob_10))
print(paste0("Predicted 15-year comorbidity-free survival probability: ", prob_15))

plotPredictSurvProb(rf.grow, newdata=testset, times=0:15)