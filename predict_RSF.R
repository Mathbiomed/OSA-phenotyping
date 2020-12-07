library(survival)
library(randomForestSRC)
library(MLmetrics)

load(file='RSF_model.rda')
#Replace the values of the pandas dataframe "random_testset" with real scaled data of the corresponding features
random_test = read.csv('random_testset.csv', row.names='index')

print(paste0("Predicted comorbidity-free window: ", predict(rf.grow, random_test)$predicted))