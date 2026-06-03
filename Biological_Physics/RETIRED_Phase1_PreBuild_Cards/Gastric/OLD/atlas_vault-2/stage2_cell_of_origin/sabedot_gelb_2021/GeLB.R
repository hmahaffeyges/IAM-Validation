library(caret)
library(randomForest)
library(parallel)
library(dplyr)

cores = parallel::detectCores()

## Wilcoxon-test

my.wilcox.test.p.value <- function(...) {
  obj<-try(wilcox.test(...), silent=TRUE)
  if (is(obj, "try-error"))
    return(NA)
  else
    return(obj$p.value)
}

### DNA methylation data (beta.anno)
# Filtered out the X and Y chromosomes and probes that were designed for sequences with known polymorphisms or probes with poor mapping quality
data <- beta.anno[!beta.anno$MASK_general & beta.anno$CpG_chrm %in% paste0("chr",1:22),]

#### STEP 1
## Selection of the discovery set. N=34 glioma primary serum samples and N=19 non-glioma serum samples
set.seed(2005518)
glioma.primary.DiscoverySet <- sample(glioma.primary.serum,size=34)

set.seed(17760704)
non.glioma.DiscoverySet <- sample(non.glioma.serum,size=19)


Prediction.Models <- (mclapply(1:1000,
                        function(i) {
                          #### STEP 2
                          ## Define the training set (N=27 gliomas; N=15 non-gliomas) and perform the differential analysis (Wilcoxon rank-sum test followed by FDR)
                          set.seed(seed = i)
                          glioma.trainingSet <- sample(glioma.primary.DiscoverySet,size=27)
                          g <- data[,glioma.trainingSet]
                          
                          set.seed(seed = i)
                          non.glioma.trainingSet <- sample(non.glioma.DiscoverySet,size=15)
                          ng <- data[,non.glioma.trainingSet]
                          
                          training.data <- cbind.data.frame(ng,g)
                          
                          p <- apply(training.data,1,function(x) {                                 
                            zz <- my.wilcox.test.p.value(as.matrix(x[1:15]),as.matrix(x[16:42]),na.action=na.omit)
                            return(zz)
                          })
                          g.m <- apply(g,1,mean,na.rm=T)
                          g.m <- g.m - Glioma.tissue$mean.DNAmeth
                          
                          p.adj <- p.adjust(p,method="fdr")
                          p.adj <- merge(p.adj,g.m,by=0,all.x=T)
                          colnames(p.adj) <- c("probeID","p.adj","Glioma.tissue.mean.DNAmeth")

                          ## Signature set
                          p <- (head(p.adj[order(p.adj[,"p.adj"], decreasing= F),], n = 1000))
                          p <- as.character(p[p$Glioma.tissue.mean.DNAmeth < 0.05 & p$Glioma.tissue.mean.DNAmeth > -0.05 & !is.na(p$Glioma.tissue.mean.DNAmeth),"probeID"])

                          p <- rownames(na.omit(data[p,c(glioma.primary.DiscoverySet,non.glioma.DiscoverySet)]))

                          training.data <- as.data.frame(t(training.data[p,]))
                          training.data$class <- c(rep("non.glioma",length(non.glioma.trainingSet)),rep("glioma",length(glioma.trainingSet)))
                          training.data$class <- factor(training.data$class)

                          #### STEP 3 
                          ## Generate Predictive Model   
                          fitControl <- trainControl(
                             method = "repeatedcv",
                             number = 10,
                             repeats = 10)
                            
                          set.seed(210)
                          mtryVals <- floor(c(seq(100, 2000, by=100),
                                              sqrt(ncol(training.data))))
                          mtryGrid <- data.frame(.mtry=mtryVals)

                          set.seed(420)

                          RF.obj <- train(class ~ .,
                                           data = training.data, 
                                           method = "rf",
                                           trControl = fitControl, 
                                           ntree = 5000, 
                                           importance = TRUE,
                                           tuneGrid = mtryGrid)
                          save(RF.obj,file=paste0("RF.",i,".rda"))

                          #### STEP 4
                          ## Test set
                          glioma.TestSet <- setdiff(glioma.primary.DiscoverySet,glioma.trainingSet)
                          non.glioma.TestSet <- setdiff(non.glioma.DiscoverySet,non.glioma.trainingSet)
                          Test.set <- data[p,c(non.glioma.TestSet,glioma.TestSet)]
                          Test.set <- as.data.frame(t(Test.set))

                          #### STEP 5
                          ## Apply Predictive model to classify test set
                          Test.set$RF.score <- predict(RF.obj,Test.set,type = "prob")$glioma
                          Test.set$class <- c(rep("non.glioma",length(non.glioma.TestSet)),rep("glioma",length(glioma.TestSet)))
                          Test.set <- Test.set[,c("RF.score","class")]

                          #### STEP 6
                          ## Store RF score
                          return(Test.set)
                        }, mc.cores=cores))

Prediction.Models.df <- bind_cols(Prediction.Models)
cols <- seq(from=1,to=ncol(Prediction.Models.df),by=2)
Prediction.Models.df <- Prediction.Models.df[,c(2,cols)]

#### STEP 7 
## Select the signature+model that correctly classifies Glioma and Non-glioma with 100% accuracy 
best.RF.model <- rbind.data.frame(aggregate(.~class, Prediction.Models.df, max)[2,],aggregate(.~class, Prediction.Models.df, min)[1,])
best.RF.model <- best.RF.model[,-1]
best.RF.model <- best.RF.model[2,] - best.RF.model[1,]
i <- which(as.numeric(best.RF.model) == max(as.numeric(best.RF.model)))

#### STEP 8
## Evaluate GeLB with an independent validation set
load(paste0("RF.",i,".rda"))

GeLB.signature <- RF.obj$trainingData
GeLB.signature <- colnames(GeLB.signature)[-ncol(GeLB.signature)]

glioma.validationSet <- setdiff(glioma.primary.serum,glioma.primary.DiscoverySet)
non.glioma.validationSet <- setdiff(non.glioma.serum,non.glioma.DiscoverySet)
Validation.set <- data[GeLB.signature,c(non.glioma.validationSet,glioma.validationSet)]
Validation.set <- as.data.frame(t(Validation.set))

Validation.set$GeLB.score <- predict(RF.obj,Validation.set,type = "prob")$glioma

#### STEP 9
## Apply GeLB using longitudinal data
Application.set <- data[GeLB.signature,glioma.recurrent.serum]
Application.set <- as.data.frame(t(Application.set))

Application.set$GeLB.score <- predict(RF.obj,Application.set,type = "prob")$glioma
