#install.packages("plot3D")
#install.packages("plotly")
#install.packages("rgl")
library(rgl)
library(plot3D)
library(plotly)
library(shiny)
library(ggplot2)
#install.packages("shiny")
library(jsonlite)
library(rjson)
setwd("D:/gproject")
json<- fromJSON(file = "nndump.json")
df<-json$training

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
  histOut = readBin(histData, integer(), n=shapeH,size=8)
  binOut = readBin(binData, double(), n=shapeB,size=4)
  binOut<- binOut[-1]
  dataOut[,1]=histOut
  dataOut[,2]=binOut
  return(dataOut)
}
###########################################################
#plot each
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
PlotAllEpoch <- function(layerI,kernelbiasI) {
  #get hist & bin_edges All
  hist<- matrix(nrow = 50,ncol = 10)
  hist<- as.data.frame(hist)
  bin_edges<- matrix(nrow = 50,ncol = 10)
  bin_edges<- as.data.frame(bin_edges)
  #df[i][6=weight][1=conv2d,2=maxpooling,3=cov2d_1,4=flatten,5=dense][1=kernel,2=bias][1=hist,2=bin]
  for(i in 1:10){
    #epoch,layer,kernel/bias
    histbin=DataFunc(i,layerI,kernelbiasI)
    hist[,i] =histbin[,1]
    bin_edges[,i] =histbin[,2]
  }
  
  df <- NULL
  for(i in 1:10){
    temp_df <- data.frame(x=bin_edges[,i], y=hist[,i], col=rep(i:i, each=10))
    df <- rbind(df,temp_df)
  }
  
  #plot
  a<-factor(df$col)
  levels(a) <- c("epoch1", "epoch2", "epoch3", "epoch4", "epoch5", "epoch6", "epoch7", "epoch8", "epoch9", "epoch10")
  plot_ly(x=df$x, y=df$y,z=df$col,color=a,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
    layout(title = 'condv kernel',
           xaxis = list(range = c(-0.5, 0.5)), 
           yaxis = list(range = c(0, 12500)))
  
}
#############################################################
#
# Main 
#df[i][6=weight][1=conv2d,2=maxpooling,3=cov2d_1,4=flatten,5=dense][1=kernel,2=bias][1=hist,2=bin]
#PlotEachFunc(1,5,1) #(epoch,layer,kernelbias)
PlotAllEpoch(1,1)
