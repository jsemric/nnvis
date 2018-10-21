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

json<- fromJSON(file = "nndump.json")
df<-json$training

hist<- matrix(nrow = 50,ncol = 10)
hist<- as.data.frame(hist)
bin_edges<- matrix(nrow = 51,ncol = 10)
bin_edges<- as.data.frame(bin_edges)
#df[i][6=weight][1=conv2d,2=maxpooling,3=cov2d_q,4=flatten,5=dense][1=kernel,2=bias][1=hist,2=bin]
for (i in 1:10){
  deHist<-base64_dec(df[[i]][[6]][[1]][[1]][[1]]$data)
  deBin<-base64_dec(df[[i]][[6]][[1]][[1]][[2]]$data)
  #decd1 = base64_dec(den1)
  shapeH<-df[[i]][[6]][[1]][[1]][[1]]$shape
  shapeB<-df[[i]][[6]][[1]][[1]][[2]]$shape
  #n=shaped
  hist[,i] = readBin(deHist, integer(), n=50,size=8)
  bin_edges[,i] = readBin(deBin, double(), n=51,size=4)
}
bin_edges<- bin_edges[-1,]
#ggplt=ggplot(data=df)
#for(i in 1:10){
#   ggplt<-ggplt+geom_line(aes(x=bin_edges[,i], y=hist[,i], colour = "red"))  
#}
#plot(ggplt)
df <- NULL
for(i in 1:10){
  temp_df <- data.frame(x=bin_edges[,i], y=hist[,i], col=rep(i:i, each=10))
  df <- rbind(df,temp_df)
}

#p<-ggplot(df,aes(x=x,y=y,group=col,colour=factor(col))) + geom_line() # plot data
#p <- p + geom_tile(aes(fill=col))
#plot(p)
#ggplotly(p)
summary(df)
factor(df$col)
a<-factor(df$col)
levels(a) <- c("epoch1", "epoch2", "epoch3", "epoch4", "epoch5", "epoch6", "epoch7", "epoch8", "epoch9", "epoch10")
#plot_ly(x=df$x, y=df$y,z=df$col,colors=colors,type = 'scatter3d', mode = 'lines')%>%
#a <- factor(c("epoch1", "epoch2", "epoch3", "epoch4", "epoch5", "epoch6", "epoch7", "epoch8", "epoch9", "epoch10"))
plot_ly(x=df$x, y=df$y,z=df$col,color=a,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
  layout(title = 'condv kernel',
    xaxis = list(range = c(-0.5, 0.5)), 
    yaxis = list(range = c(0, 12500)))

#+ xlim(-0.5, 0.5) + ylim(0, 12500)
#lines3D(df$x, df$y, df$col,ticktype = "detailed")


