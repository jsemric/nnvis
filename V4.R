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
json<- fromJSON(file = "nndump.json")
df<-json$training
img<-json[[2]]$image_data

###############################################
#function get data
DataFunc <- function(epoch,layer,kernelbias) {
  histOut<- matrix(nrow = 50,ncol = 1)
  binOut<- matrix(nrow = 51,ncol = 1)
  dataOut<- matrix(nrow = 50,ncol = 2)
  histData<-base64_dec(df[[epoch]][[6]][[layer]][[kernelbias]][[1]]$data)
  binData<-base64_dec(df[[epoch]][[6]][[layer]][[kernelbias]][[2]]$data)
  shapeH<-df[[epoch]][[6]][[layer]][[kernelbias]][[1]]$shape
  shapeB<-df[[epoch]][[6]][[layer]][[kernelbias]][[2]]$shape
  histOut = readBin(histData, double(), n=shapeH,size=4)
  binOut = readBin(binData, double(), n=shapeB,size=4)
  binOut<- binOut[-1]
  dataOut[,1]=histOut
  dataOut[,2]=binOut
  return(dataOut)
}
###########################################################
#plot each
#df[epoch][6=weight][1=conv2d,3=cov2d_1,6=dense][1=kernel,2=bias][1=hist,2=bin]
#PlotEachFunc(1,5,1) #(epoch,layer,kernelbias)
PlotEachFunc <- function(epochI,layerI,kernelbiasI) {
  dataO=DataFunc(epochI,layerI,kernelbiasI)
  histO=dataO[,1]
  binO=dataO[,2]
  plot_ly(x=binO, y=histO,z=epochI,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
    layout(title = 'condv kernel',
           xaxis = list(range = c(-0.5, 0.5)), 
           yaxis = list(range = c(0, 12500)))
}
###########################################################
#data for plot All epoch
#[1=conv2d,3=cov2d_1,6=dense][1=kernel,2=bias]
PlotAllEpoch <- function(layerI,kernelbiasI) {
  #get hist & bin_edges All
  epoch=length(df)
  hist<- matrix(nrow = 50,ncol = epoch)
  hist<- as.data.frame(hist)
  bin_edges<- matrix(nrow = 50,ncol = epoch)
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
           xaxis = list(range = c(-0.5, 0.5)), 
           yaxis = list(range = c(0, 12500)))
  
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
  
  plot_ly(x=val$epoch, y=val$value ,mode = 'lines',line=list(width=5))    
}  

###########################################################
#data for plot some epoch
#inEpoch <- c(TRUE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)
#PlotSomeEpoch(inEpoch,1,1) #(epochI,layerI,kernelbiasI) 
PlotSomeEpoch <- function(epochI,layerI,kernelbiasI) {
  epoch=length(df)
  trueCount=sum(epochI, na.rm = TRUE)
  hist<- matrix(nrow = 50,ncol = trueCount)
  hist<- as.data.frame(hist)
  bin_edges<- matrix(nrow = 50,ncol = trueCount)
  bin_edges<- as.data.frame(bin_edges)
  k<-1
  for(i in 1:epoch){
    if(epochI[i]==TRUE){
      histbin=DataFunc(i,layerI,kernelbiasI)
      hist[,k] =histbin[,1]
      bin_edges[,k] =histbin[,2]
      k<-k+1
    }
  }
  
  df1 <- NULL
  if(trueCount!=0){
    
    for(i in 1:trueCount){
      temp_df <- data.frame(x=bin_edges[,i], y=hist[,i], col=i)
      df1 <- rbind(df1,temp_df)
    }
  }
  
  
  #plot
  a<-factor(df1$col)
  list=which(inEpoch==TRUE)
  levelEpoch=paste("epoch",list )
  levels(a) <- c(levelEpoch)
  plot_ly(x=df1$x, y=df1$y,z=df1$col,color=a,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
    layout(title = 'condv kernel',
           xaxis = list(range = c(-0.5, 0.5)), 
           yaxis = list(range = c(0, 12500)))
}  
#############################################################
#
#input image
#
#########
#
PlotInputImg<- function() {
  imgIn=img$input_data
  inShape=imgIn$shape
  inData<-base64_dec(imgIn$data)
  inOut = readBin(inData, double(), n=inShape[1]*inShape[2]*inShape[3]*inShape[4],size=4)
  inOut =array_reshape(inOut,inShape) #library(reticulate)
  require(grDevices)
  library("imager")
  library(reticulate)
  par(mfrow=c(1,inShape[1]))  
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
#
#########
#

#
#
#############################################################
#
# Main 
#df[i][6=weight][1=conv2d,3=cov2d_1,6=dense][1=kernel,2=bias][1=hist,2=bin]
#PlotEachFunc(1,5,1) #(epoch,layer,kernelbias)
PlotAllEpoch(1,1) #(layerI,kernelbiasI)
#inEpoch <- c(TRUE,FALSE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE,FALSE)
#PlotSomeEpoch(inEpoch,1,1) #(epochI,layerI,kernelbiasI) 
#
#[2=acc,3=loss,4=val_loss,5=val_acc]
#PlotScalars(3) #(graphI)
#
#image
#PlotInputImg()
