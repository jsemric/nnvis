source("V4.R")




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
                                                  choices  = c("dense","conv2d_1","conv2d")),
                                      selectInput("layer2","select kernel or bias",
                                                  choices = c("kernel","bias"))),
                         mainPanel(
                           textOutput("hour1"),
                           plotlyOutput("Scatter1"))
                       )
              ),
              
              ###################### ######################### ############################2015#################
              tabPanel(title = "Imagines",
                       tabsetPanel(type = "tab",
                                   tabPanel("Input",
                                            verbatimTextOutput("hour2"),
                                            plotOutput("Scatter2")),
                                   tabPanel("Output",
                                            sidebarLayout(
                                              sidebarPanel(("select yo shits"),
                                                           selectInput("piclayer", "select  a layer",
                                                                       choices  = c("conv2d_1","conv2d")),
                                                           selectInput("pic","select YOUR fuckingSHIT",
                                                                       choices = 1:numIng),
                                                           conditionalPanel(condition="$('html').hasClass('shiny-busy')",
                                                                            tags$div("Loading...",id="loadmessage")
                                                           )
                                                           
                                            ),
                                            mainPanel(
                                              textOutput("haha2"),
                                              plotOutput("plotoutput")
                                            ))
                                            
                                            )
                                   )),
                       
              
              ############################################   2016 ################### ######################################################
               tabPanel(title = "Diff",
                        sidebarLayout(
                          sidebarPanel((""),
                                       selectInput("layer3","select layer",
                                                   choices  = c("dense","conv2d_1","conv2d")),
                                       #selectInput("layer1","select YOUR fuckingSHIT",choices = c("1"=1,"2"=2,"3"=3,"4"=4,"5"=5,"6"=6,"7"=7,"8"=8,"9"=9,"10"=10)),
                                       selectInput("layer4","select kernel or bias",
                                                   choices = c("kernel","bias"))),
                          mainPanel(
                            textOutput("hour3"),
                            plotlyOutput("Scatter3"))
                        )
              
               ),
              
              tabPanel(title = "pca",
                       verbatimTextOutput("hour4"),
                       
                       plotlyOutput("Scatter4"))
              
              ))






server<- function(input,output){
  
  ##################################14444444444-----------7777777777777777
  output$hour<-renderText({
    "loss acc "
  })
  
  
  output$Scatter<- renderPlot({
    PlotScalars()
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
  
  output$haha2<-renderText({
    paste(" ",input$pic,input$piclayer)
  })
  
  output$Scatter1<- renderPlotly({
    
    kb=input$layer2
    kb=paste(kb,":0",sep='')
    kb=paste("/",kb,sep='')
    kb=paste(input$layer,kb,sep='')
    PlotAllEpoch(input$layer,kb)
    
  })

  #######################################Hour
  
  
  output$hour2<-renderText({
    "Input image"
  })
  
  
  output$Scatter2<- renderPlot({
    PlotInputImg()
    
  }, height = 500, width = 800)
  
  output$plotoutput<- renderPlot({
    PlotOutputImg(input$piclayer,as.integer(input$pic))  }, height =500, width = 500)
  
  # output$hour3<-renderText({
  #  "
  # BABYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
  #  })
  
  output$Scatter3<- renderPlotly({
    kb=input$layer4
    kb=paste(kb,":0",sep='')
    kb=paste("/",kb,sep='')
    kb=paste(input$layer3,kb,sep='')
    PlotMADiff(input$layer3,kb)
  })
  
  output$Scatter4<- renderPlotly({
    
    PlotVal()
  })
  
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


