load<- c("ggmap","ggplot2","dplyr","gtools","readr","plyr","lubridate","MASS","caret"
         ,"mgcv","nlme","gridExtra","qpcR","ggpubr",
         "stplanr","ggrepel","forcats","shiny","plotrix","rsconnect","sjPlot","dplyr",
         "plotly")
#install.packages('rsconnect')
options(scipen=999)
sapply(load, require, character = TRUE)
memory.limit(size = 50000)
library(sjPlot,lib.loc="P:/R-3.3.3./library-all/")
library(dplyr,lib.loc="P:/R-3.3.3./library-all/")
library(plotly,lib.loc="P:/R-3.3.3./library-all/")
library(shiny,lib.loc="P:/R-3.3.3./library-all/")



data = list.files(pattern="*.csv")
list2env(
  lapply(setNames(data, make.names(gsub("*.csv$", "", data))), 
         read.csv), envir = .GlobalEnv)
x<- 1:10
y<- 1:10

###########################################plots functions 
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
img<-json[[2]]$image_data
epochloss<- read.table("scatter11.csv",sep = ",",header = T)
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
#PlotAllEpoch(1,1)
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
#PlotInputImg()
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
#[1=conv2d,3=cov2d_1,6=dense]
#PlotOutputImg(1)
#########
#

#
#
#############################################################
#
# Main 
#df[i][6=weight][1=conv2d,3=cov2d_1,5=dense][1=kernel,2=bias][1=hist,2=bin]
#PlotEachFunc(1,5,1) #(epoch,layer,kernelbias)

############################################################################
################################################################


#################################################################################################
ui<- fluidPage(
  tags$head(
    tags$style(HTML("
                    @import url('//fonts.googleapis.com/css?family=Lobster|Cabin:400,700');
                    "))
    ),
  titlePanel(title= h1("NN visualization",
                                    style = "font-family: 'Lobster', cursive;
        font-weight: 500; line-height: 1.1; 
                                    color: #4d3a7d;",align="center")),
             
               ###########frist tab panel 
               tabsetPanel(type = "pills" ,
                           ################################    overall 
                           #################################              ######################################
                           tabPanel("Scalars",
                                                verbatimTextOutput("hour"),
                                    
                                    
                                                 plotOutput("Scatter"),
                                    tags$style("#week {font-size:15px;
                                               bottom: 100px; 
                                               position:bottom;
                                               width: 100%;
                                               left:0px;}"),
                                    div(style="text-align:bottom;
                                        padding-top:70px;
                                        position:relative;",
                                        textOutput("week")
                                    )
                                                
                           ),
                           
                           
                           
                           tabPanel("Histograms",
                                    
                                    sidebarLayout(
                                      sidebarPanel(("whtdasgdasgdajs"),
                                                   selectInput("layer","select layer",
                                                               choices  = c("dense"=6,"conv2d_1"=3,"conv2d"=1)),
                                                   #selectInput("layer1","select YOUR fuckingSHIT",choices = c("1"=1,"2"=2,"3"=3,"4"=4,"5"=5,"6"=6,"7"=7,"8"=8,"9"=9,"10"=10)),
                                                   selectInput("layer2","select kernel or bias",
                                                               choices = c("kernel"=1,"bias"=2))),
                                    mainPanel(
                                    textOutput("hour1"),
                                    plotlyOutput("Scatter1"))
                                    )
                                    ),
                                    
                           ###################### ######################### ############################2015#################
                           tabPanel(title = "Imagines",
                                    verbatimTextOutput("hour2"),
                                    
                                    plotOutput("Scatter2"))#,
                                    
                           ############################################   2016 ################### ######################################################
                          # tabPanel(title = "Distributions",
                           #         verbatimTextOutput("hour3"),
                                    
                                   # plotOutput("Scatter3"))
                              
               ))






server<- function(input,output){
  
  ##################################14444444444-----------7777777777777777
  output$hour<-renderText({
    "loss acc "
  })
  
  
  output$Scatter<- renderPlot({
    #plot scalar
    #[2=acc,3=loss,4=val_loss,5=val_acc]
    #PlotScalars(3) #(graphI)
    z<- ggplot(epochloss, aes(x=epoch,y=loss)) + geom_line()+
      scale_x_discrete(limits=seq(0,9))+geom_point()+theme_dark()
    
    z1<-ggplot(epochloss, aes(x=epoch,y=acc)) + geom_line()+
      scale_x_discrete(limits=seq(0,9))+geom_point()
    
    z2<-ggplot(epochloss, aes(x=epoch,y=auc)) + geom_line()+
      scale_x_discrete(limits=seq(0,9))+geom_point()
    
    z3<-ggplot(epochloss, aes(x=epoch,y=val_auc)) + geom_line()+
      scale_x_discrete(limits=seq(0,9))+geom_point()
    
    grid.arrange(z,z1,z2,z3,nrow=2,ncol=2)
    
    
  })
  
  output$week <- renderText("
                            
1
2
2
                            x1 prev weekkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
  
  
 # output$dayyear<- renderPlotly({
#    ggplotly(ggplot(weather,aes(x=day,y=freq,col=factor(year)))+geom_line(alpha=0.5,lwd=0.3),labs(col="Year"))
#  })
  
 # output$cmax<- renderText({
    
 #   "Say something"
    
 # })
  
  output$hour1<-renderText({
    paste(" ",input$layer,input$layer1,input$layer2)
  })
  
  
  output$Scatter1<- renderPlotly({
    
    PlotEachFunc <- function(epochI,layerI,kernelbiasI) {
      dataO=DataFunc(epochI,layerI,kernelbiasI)
      histO=dataO[,1]
      binO=dataO[,2]
      plot_ly(x=binO, y=histO,z=epochI,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
        layout(title = 'condv kernel',
               xaxis = list(range = c(-0.5, 0.5)), 
               yaxis = list(range = c(0, 12500)))
    }
    #PlotEachFunc(1,1,1)
    
    #PlotEachFunc(as.integer(input$layer1),as.integer(input$layer),as.integer(input$layer2))
    PlotAllFunc <- function(layerI,kernelbiasI) {
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
    PlotAllFunc(as.integer(input$layer),as.integer(input$layer2))
    
  })
  
  
  #######################################Hour

  
  output$hour2<-renderText({
    "Input image"
  })
  
  
  output$Scatter2<- renderPlot({
    PlotInputImg()
    
  }, height = 500, width = 800)


 # output$hour3<-renderText({
  #  "
    
    
    
    
    
    
    
   # BABYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
#  })
  
#output$Scatter3<- renderPlot({
#  plot((x^4),y)
  
#}, height = 500, width = 800)

}

shinyApp(ui,server)



#library(rsconnect)
#library(Rcmdr)
#deployApp()


ui <- shinyUI(
  fluidPage(
    tags$style("#x1_current_week {font-size:20px;
               color:red;
               display:block; }"),
    tags$style("#x1_previous_week {font-size:15px;
               display:block;
               bottom: 12px; 
               position:absolute;
               width: 100%;
               left:0px;}"),
    div(style="text-align:center;
        box-shadow: 10px 10px 5px #888888;
        width:200px;
        height:200px;
        padding-top:70px;
        position:relative;",
        textOutput("x1_current_week"),
        textOutput("x1_previous_week")
    )
    )
  )

server <- function(input,output)
{
  output$x1_current_week  <- renderText("x1 this week")
  output$x1_previous_week <- renderText("x1 prev week")
  
}


shinyApp(ui,server)

