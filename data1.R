load<- c("ggmap","ggplot2","dplyr","gtools","readr","plyr","lubridate","MASS","caret"
         ,"mgcv","nlme","gridExtra","qpcR","ggpubr",
         "stplanr","ggrepel","forcats","shiny","plotrix","rsconnect","sjPlot","dplyr",
         "plotly")
#install.packages('rsconnect')
options(scipen=999)
sapply(load, require, character = TRUE)
memory.limit(size = 50000)
library(sjPlot)
library(dplyr)
library(plotly)
library(shiny)
#install.packages("shiny")
library(jsonlite)
library(rjson)

haha <- fromJSON(file = 
                  "myfile.json")


str(haha)
lapply(haha, function (x) x[c('training')])
#haha1<- as.data.frame(haha[[1]])
#haha2<- as.data.frame(haha[[2]][[1]][[3]])
x<- 1:10

for (i in 1:10){
  x[i]<-haha[[2]][[i]][[3]] 
}
epochloss<- as.data.frame(0:9)


epochloss[,2]<- x
colnames(epochloss)<-c("epoch","acc","loss","auc","val_auc")

for (i in 1:10){
  epochloss[i,5]<-haha[[2]][[i]][[5]] 
}


haha[[2]][[2]][[1]] 
#################################################

# plot loss
z<- ggplot(epochloss, aes(x=epoch,y=loss)) + geom_line()+
  scale_x_discrete(limits=seq(0,9))+geom_point()

z1<-ggplot(epochloss, aes(x=epoch,y=acc)) + geom_line()+
  scale_x_discrete(limits=seq(0,9))+geom_point()

z2<-ggplot(epochloss, aes(x=epoch,y=auc)) + geom_line()+
  scale_x_discrete(limits=seq(0,9))+geom_point()

z3<-ggplot(epochloss, aes(x=epoch,y=val_auc)) + geom_line()+
  scale_x_discrete(limits=seq(0,9))+geom_point()

grid.arrange(z,z1,z2,z3,nrow=2,ncol=2)

# extract weights from the first layer in 3rd epoch
json = fromJSON("myfile.json")
haha<- json
layers = json$layers
df = json$training[[1]]
# plot loss
ggplot(df, aes(x=epoch,y=loss)) + geom_line()

# extract weights from the first layer in 3rd epoch
epoch = 3
kernel = df$weights$dense["dense/kernel:0"]
bias = df$weights$dense['dense/bias:0']
rawdata = kernel[epoch,]$data
shape = kernel[epoch,]$shape[[1]]
n = cumprod(shape)[[-1]] # number of elements in tensor
# decode, convert and reshape
dec = base64_dec(rawdata)
kernel_weights = readBin(dec, double(), n=n, size=4)
dim(kernel_weights) = shape
