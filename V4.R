#install.packages("plot3D")
#install.packages("plotly")
#install.packages("rgl")
#library(rgl)
#library(plot3D)
library(plotly)
library(shiny)
library(ggplot2)
#install.packages("shiny")
library(jsonlite)
library(rjson)
library(reticulate)
require(grDevices)
library("imager")
library(reticulate)
json<- fromJSON(file = "nndump.json")
df<-json$training
img<-json$train_end$image_data
val<-json$train_end$validation_data
###############################################
#function get data
DataFunc <- function(epoch,layer,kernelbias) {
  shaH<-df[[1]][[6]][[layer]][[kernelbias]][[1]]$shape
  shaB<-df[[1]][[6]][[layer]][[kernelbias]][[2]]$shape
  histOut<- matrix(nrow = shaH,ncol = 1)
  binOut<- matrix(nrow = shaB,ncol = 1)
  dataOut<- matrix(nrow = shaH,ncol = 2)
  histData<-base64_dec(df[[epoch]][[6]][[layer]][[kernelbias]][[1]]$data)
  binData<-base64_dec(df[[epoch]][[6]][[layer]][[kernelbias]][[2]]$data)
  shapeH<-df[[epoch]][[6]][[layer]][[kernelbias]][[1]]$shape
  shapeB<-df[[epoch]][[6]][[layer]][[kernelbias]][[2]]$shape
  histOut = readBin(histData, double(), n=shapeH,size=4)
  binOut = readBin(binData, double(), n=shapeB,size=4)
  binOut<- binOut[-1]
  dataOut[,1]=histOut
  dataOut[,2]=binOut
  print(dataOut)
  return(dataOut)
}

###########################################################
#data for plot All epoch
#[1=conv2d,3=cov2d_1,6=dense][1=kernel,2=bias]
#PlotAllEpoch(1,1)
PlotAllEpoch <- function(layerI,kernelbiasI) {
  #get hist & bin_edges All
  epoch=length(df)
  shape<-df[[1]][[6]][[layerI]][[kernelbiasI]][[1]]$shape
  hist<- matrix(nrow = shape,ncol = epoch)
  hist<- as.data.frame(hist)
  bin_edges<- matrix(nrow = shape,ncol = epoch)
  bin_edges<- as.data.frame(bin_edges)
  #df[i][6=weight][1=conv2d,2=maxpooling,3=cov2d_1,4=flatten,5=dense][1=kernel,2=bias][1=hist,2=bin]
  for(i in 1:epoch){
    #epoch,layer,kernel/bias
    histbin=DataFunc(i,layerI,kernelbiasI)
    hist[,i] =histbin[,1]
    bin_edges[,i] =histbin[,2]
  }
  
  df1 <- NULL
  for(i in 1:epoch){
    temp_df <- data.frame(x=bin_edges[,i], y=hist[,i], col=i)
    df1 <- rbind(df1,temp_df)
  }
  
  #plot
  a<-factor(df1$col)
  levelEpoch=paste("epoch",1:epoch)
  levels(a) <- c(levelEpoch)
  #levels(a) <- c("epoch1", "epoch2", "epoch3", "epoch4", "epoch5", "epoch6", "epoch7", "epoch8", "epoch9", "epoch10")
  plot_ly(x=df1$x, y=df1$y,z=df1$col,color=a,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
    layout(title = '',
           xaxis = list(range = c(min(df1$x), max(df1$x))), 
           yaxis = list(range = c(min(df1$y), max(df1$y))))
  
}

###########################################################
#plot scalar
#[2=acc,3=loss,4=val_loss,5=val_acc]
#PlotScalars(3) #(graphI)
PlotScalars <- function(graphI) {
  epoch=length(df)
  val<- matrix(nrow = epoch,ncol = 2)
  val<- as.data.frame(val)
  names(val) <- c("epoch", "value")
  for(i in 1:epoch){
    val[i,1]=i
    val[i,2]=df[[i]][[graphI]]
  }
  
  return (plot_ly(x=val$epoch, y=val$value ,mode = 'lines',line=list(width=5))  )  
}  


#############################################################
#
#input image
#PlotInputImg()
#########
#
PlotInputImg<- function() {
  imgIn=img$input_data
  inShape=imgIn$shape
  inData<-base64_dec(imgIn$data)
  inOut = readBin(inData, double(), n=prod(inShape),size=4)
  inOut =array_reshape(inOut,inShape) #library(reticulate)
  require(grDevices)
  library("imager")
  library(reticulate)
  par(mfrow=c(inShape[1]/5,5))  
  for(i in 1:inShape[1]){
    im<-inOut[i,,,]
    getIm<-as.cimg(im)
    plot(getIm,axes=FALSE)
  }
}
#
#############################################################
#
#Output image
#lastFill-firstFill no more than 10 no more than 10 img each 
#[1=conv2d,2s=cov2d_1]
#PlotOutputImg(1,11,10) #(layerI,imgI,firstFill,lastFill) 
#########
#
PlotOutputImg<- function(layerI,imgI,firstFill,lastFill) {
  imgOut=img$outputs[[layerI]]
  outShape=imgOut$shape
  outData<-base64_dec(imgOut$data)
  outOut = readBin(outData, double(), n=prod(outShape),size=4)
  outOut =array_reshape(outOut,outShape) #library(reticulate)
  par(mfrow=c(ceiling((lastFill-firstFill)/5),5))  
  for(i in firstFill:lastFill){
    im<-outOut[imgI,,,i]
    getIm<-as.cimg(im)
    plot(getIm,axes=FALSE)
  }
}
##################################################
# plot val

PlotVal<- function() {
  labels = val$labels
  predictions = val$predictions
  val_data = val$val_data
  print(val_data$shape)
  
  
  labelsData<-base64_dec(labels$data)
  shapelabels<-labels$shape
  labelOut = readBin(labelsData, double(), n=prod(shapelabels),size=4)
  labelOut =array_reshape(labelOut,shapelabels)
  
  valData<-base64_dec(val_data$data)
  shapeval<-val_data$shape
  valOut = readBin(valData, double(), n=prod(shapeval),size=4)
  valOut =array_reshape(valOut,shapeval)
  
  getVal =array_reshape(valOut,c(shapeval[1],-1))
  pca <- princomp(valOut, scores=T, cor=T)
  pca=array_reshape(pca$score,c(shapeval[1],-1))
  a <- as.factor(labelOut)
  plot_ly(x=pca[,1], y=pca[,2],z=pca[,3],type = 'scatter3d',color=a ,alpha=0.6,size=0.5)
  
}




#
#
#############################################################
#
# Main 
#df[epoch][6=weight][1=conv2d,3=cov2d_1,6=dense][1=kernel,2=bias][1=hist,2=bin]
#PlotEachFunc(1,5,1) #(epoch,layer,kernelbias)
#PlotAllEpoch(1,1) #(layerI,kernelbiasI)

#inEpoch <- c(TRUE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)
#PlotSomeEpoch(inEpoch,1,1) #(epochI,layerI,kernelbiasI) 

#[2=acc,3=loss,4=val_loss,5=val_acc]
#PlotScalars(3) #(graphI)

#image
#PlotInputImg()

#[1=conv2d,2=cov2d_1]
#lastFill-firstFill no more than 10 
#PlotOutputImg(1,4,1,10) #(layerI,imgI,firstFill,lastFill) 

