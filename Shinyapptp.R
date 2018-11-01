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
  
  df <- NULL
  for(i in 1:epoch){
    temp_df <- data.frame(x=bin_edges[,i], y=hist[,i], col=rep(i:i, each=epoch))
    df <- rbind(df,temp_df)
  }
  
  #plot
  a<-factor(df$col)
  levelEpoch=paste("epoch",1:epoch)
  levels(a) <- c(levelEpoch)
  #levels(a) <- c("epoch1", "epoch2", "epoch3", "epoch4", "epoch5", "epoch6", "epoch7", "epoch8", "epoch9", "epoch10")
  plot_ly(x=df$x, y=df$y,z=df$col,color=a,type = 'scatter3d' ,mode = 'lines',line=list(width=5))%>%
    layout(title = 'condv kernel',
           xaxis = list(range = c(-0.5, 0.5)), 
           yaxis = list(range = c(0, 12500)))
  
}
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
  titlePanel(title= h1("SEXY AS FUCK",
                                    style = "font-family: 'Lobster', cursive;
        font-weight: 500; line-height: 1.1; 
                                    color: #4d3a7d;",align="center")),
             
               ###########frist tab panel 
               tabsetPanel(type = "pills" ,
                           ################################    overall 
                           #################################              ######################################
                           tabPanel("Graphys",
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
                                                   selectInput("layer","select YOUR SHIT",
                                                               choices  = c("dense"=5,"conv2d_1"=3,"conv2d"=1)),
                                                   selectInput("layer1","select YOUR fuckingSHIT",choices = c("1"=1,"2"=2,"3"=3,"4"=4,"5"=5,"6"=6,"7"=7,"8"=8,"9"=9,"10"=10)),
                                                   selectInput("layer2","select YOUR pwiceifSHIT",
                                                               choices = c("kernel"=1,"bias"=2))),
                                    mainPanel(
                                    textOutput("hour1"),
                                    plotlyOutput("Scatter1"))
                                    )
                                    ),
                                    
                           ###################### ######################### ############################2015#################
                           tabPanel(title = "Imagines",
                                    verbatimTextOutput("hour2"),
                                    
                                    plotOutput("Scatter2")),
                                    
                           ############################################   2016 ################### ######################################################
                           tabPanel(title = "Distributions",
                                    verbatimTextOutput("hour3"),
                                    
                                    plotOutput("Scatter3"))
                              
               ))






server<- function(input,output){
  
  ##################################14444444444-----------7777777777777777
  output$hour<-renderText({
    "4 plots of elops"
  })
  
  
  output$Scatter<- renderPlot({
      plot(1:10,1:10)
    
  }, height = 500, width = 800)
  
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
    paste("this  is wut",input$layer,input$layer1,input$layer2)
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
    
    PlotEachFunc(as.integer(input$layer1),as.integer(input$layer),as.integer(input$layer2))
    
  })
  
  
  #######################################Hour

  
  output$hour2<-renderText({
    "OH YEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
  })
  
  
  output$Scatter2<- renderPlot({
    plot((x),(y^3))
    
  }, height = 500, width = 800)


  output$hour3<-renderText({
    "
    
    
    
    
    
    
    
    BABYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
  })
  
output$Scatter3<- renderPlot({
  plot((x^4),y)
  
}, height = 500, width = 800)

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

