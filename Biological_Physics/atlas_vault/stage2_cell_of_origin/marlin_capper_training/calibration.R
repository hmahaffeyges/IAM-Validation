#-----------------------------------------------------------------------------------
# this script fits for each CV fold a multinomial L2-penalized glmnet model to calibrate RF scores
# and one final calibration model using RF scores generated in the out loop of the CV 
# After calibration CV results are calculated in CVresults.Rmd which is compiled to an html report
#
# Martin Sill
# m.sill@dkfz.de                                                                  
# 
# 2018-03-14 UTC
#------------------------------------------------------------------------------------                   
#options(max.print = 1000)
#options(stringsAsFactors = FALSE)
#options(scipen = 999)

library(rmarkdown)
library(glmnet)
library(doParallel)
library(HandTill2001)

cores <- 4

registerDoParallel(cores)

message("loading data ...",Sys.time())
load(file.path("results","Mset_filtered.RData"))
load(file.path("CV","nfold.RData"))

for(i in 1:length(nfolds)){
  scores <- list() 
  idx <- list()
  for(j in 1:length(nfolds)){
    fname <- paste0("CVfold.",i,".",j,".RData")
    load(file.path("CV",fname))
    scores[[j]] <- rf.scores
    idx[[j]] <- nfolds[[i]][[2]][[j]]$test
  }
  scores <- do.call(rbind,scores)
  idx <- unlist(idx)
  y <- anno$`methylation class:ch1`[idx]         
  
  message("fitting calbriation model fold ",i," ...",Sys.time())
  # fit multinomial logistic ridge regression model
  suppressWarnings(cv.calfit <- cv.glmnet(y=y,x=scores,family="multinomial",type.measure="mse",
                                          alpha=0,nlambda=100,lambda.min.ratio=10^-6,parallel=TRUE))
  
  fname <- paste0("CVfold.",i,".",0,".RData")
  load(file.path("CV",fname))
  
  message("calibrating raw scores fold ",i," ...",Sys.time())
  probs <- predict(cv.calfit$glmnet.fit,newx=rf.scores,type="response"
                   ,s=cv.calfit$lambda.1se)[,,1] # use lambda estimated by 10fold CVlambda
  
  
  err <- sum(colnames(probs)[apply(probs,1,which.max)] != anno$`methylation class:ch1`[nfolds[[i]][[1]][[1]]$test])/length(nfolds[[i]][[1]][[1]]$test)
  
  message("misclassification error: ",err)
  
  fname_probs <- paste0("probsCVfold.",i,".",0,".RData")
  save(probs,file=file.path("CV",fname_probs))
}

scores <- list()
idx <- list()
for(i in 1:length(nfolds)){
  fname <- paste0("CVfold.",i,".",0,".RData")
  load(file.path("CV",fname))
  scores[[i]] <- rf.scores
  idx[[i]] <- nfolds[[i]][[1]][[1]]$test
}
scores <- do.call(rbind,scores)

probl <- list()
for(i in 1:length(nfolds)){
  fname <- paste0("probsCVfold.",i,".",0,".RData")
  load(file.path("CV",fname))
  probl[[i]] <- probs
}
probs <- do.call(rbind,probl)


idx <- unlist(idx)
y <- anno$`methylation class:ch1`[idx] 

ys <- colnames(scores)[apply(scores,1,which.max)]
yp <- colnames(probs)[apply(probs,1,which.max)]

errs <- sum(y!=ys)/length(y)
errp <- sum(y!=yp)/length(y)

message("overall misclassification error scores: ",errs)
message("overall misclassification error calibrated: ",errp)

message("fitting final calibration model ...",Sys.time())

suppressWarnings(cv.calfit <- cv.glmnet(y=y,x=scores,family="multinomial",type.measure="mse",
                                        alpha=0,nlambda=100,lambda.min.ratio=10^-6,parallel=TRUE))

save(cv.calfit,file=file.path("results","calfit.RData"))

save(scores,probs,y,ys,yp,file=file.path("results","CVresults.RData"))

message("generating report ...",Sys.time())
rmarkdown::render("CVresults.Rmd")
message("finished ...",Sys.time())
