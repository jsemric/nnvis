library(shiny)
library(ggplot2)
#install.packages("ggExtra")
library(ggExtra)

lvh<- read.table("lvh.dat",header = T)
#lvh <- data.frame(lapply(lvh, function(x) as.numeric(x)))
lvh$sbp <- as.numeric(lvh$sbp)
lvh$sex <- as.integer(lvh$sex)
lvh$card <- as.integer(lvh$card)
lvh$dgp <- as.integer(lvh$dgp)
lvh$surv <- as.integer(lvh$surv)
lvh$sex <- lvh$sex - 1
lvh$card <- lvh$card - 1
lvh$dgp <- lvh$dgp - 1

attach(lvh)
######starting shiny 
####### question 1
ui<- fluidPage(titlePanel(title= h1("Left ventricular hypertrophy analysis", align="center")),
               ###########frist tab panel 
               tabsetPanel(type = "pills" ,
                           tabPanel(title = "General Statistic and histogram for questions 1 and 2",
                           titlePanel(title= h4("General Statistic", align="center")),
                           sidebarLayout(
                             
                             ##side panel for single variable information
                             sidebarPanel(("General Statistics for a Variable"),
                                          ##variable panel
                                          selectInput("var","select a variable for summary statistic", choices=c("llvmi","sbp","sex","card","dgp","surv")),
                                          selectInput("var0","select another variable to form a marginal histogram with the one selected above",
                                                      choices=c("llvmi","sbp","sex","card","dgp","surv")),
                                          #bins
                                          sliderInput("bins","Select the number of bins for the histogram",min=5,max=50,value=10)),
                             
                             ###main panel
                             mainPanel(
                               tabsetPanel(type="tab",
                                           ##summary statistic  question 1
                                           tabPanel("Summary",verbatimTextOutput("sum")),
                                           ##histogram question 2
                                           tabPanel("Histogram",plotOutput("thehist"))
                               )))
                           ),
                           tabPanel("Data information",
                                    p("
Patients receiving treatment for severe kidney disease sometimes also show signs
of an enlargement of the heart muscle, known as left ventricular hypertrophy.
                                      A study was carried out in a group of kidney patients to identify factors
                                      which might be associated with left ventricular hypertrophy and also to identify
                                      whether this condition was a risk factor for early death of the patient. The
                                      patients in the study were all adults between 19 and 50 years of age. The size
                                      of the heart muscle was assessed through measurement of the left ventricular
                                      mass index. All patients were then followed up for at least two years, or until
                                      death if that occurred earlier."),
                                    hr(),
                                    p(strong("llvmi"), ": left ventricular mass index, on a log scale"),
                                    p(strong("sbp"), ": systolic blood pressure"),
                                    p(strong("gender"), ": 1 for males, 2 for females"),
                                    p(strong("card"), ": previous cardiac disease (1 - no, 2 - yes)"),
                                    p(strong("dgp"), ": type of kidney disease (two different groups)"),
                                    p(strong("surv"), ": patient death within 2 years (0 - no, 1 - yes)")
                                    )
                           
                           ),
               hr(),
               br(),
               
               tabsetPanel(type = "pills" ,
                           tabPanel(title = "Boxplot & Densityplot for questions 3 and 4",
                           titlePanel(title= h4("Boxplot & Densityplot", align="center")),
                           
                           
                           ##start another chunk of panel to make pairwise plots easier 
                           sidebarLayout(
                             
                             sidebarPanel(("Variables selection for plots"),
                                          ##variable panel
                                          #selectInput("var1","Select the first variable",choices = c("Llvmi","SBP")),
                                          
                                          #####question 3-4 plots
                                          helpText("Please select two variables for box and density plots"),
                                          uiOutput("vx"),
                                          uiOutput("vy")
                                          
                             ),
                             
                             mainPanel(
                               tabsetPanel(type="tab",
                                           ####boxplot
                                           tabPanel("Boxplots",plotOutput("thebox")),
                                           ##density
                                           tabPanel("Densityplots",plotOutput("pairs"))
                                           ##model
                               )))
                           ),
                           tabPanel(title = "Plots information",
                                    p("For the boxplot: The (X) Variable selection contains the 
                                      categorical variables which is ploted against the continuous variables
                                      in (Y) selections "),
                                    br(),
                                    p("For the density plot, The density plot of the (Y)variable selected is
                                      plotted and grouped by the (X)variable selected"))
                           
                           ),
               hr(),
               br(),
               
               tabsetPanel(type = "pills" ,
                           tabPanel(title = "Regression for questions 5 and 6",
                           titlePanel(title= h4("Regression", align="center")),
                           
                           sidebarLayout(
                             
                             sidebarPanel(("Variables selection for Regression models"),
                                          
                                          #question 5
                                          helpText(strong("Please select the covariate for the regression model where llvmi is used as the response")),br(),
                                          uiOutput("cov"),
                                          ##question 6
                                          helpText(strong("For the following, please select a response,variables and the regression model")),hr(),
                                          uiOutput("response"),
                                          
                                          uiOutput("cov1"),
                                          
                                          uiOutput("reg")
                             ),
                             
                             
                             mainPanel(
                               tabsetPanel(type="tab",
                                           ##model
                                           tabPanel("Modelling llvmi",
                                                    verbatimTextOutput("sum1"),
                                                    verbatimTextOutput("ANOVA"),
                                                    plotOutput("assumptions")),
                                           
                                           tabPanel("Regression modelling",
                                                    verbatimTextOutput("sum2"),
                                                    verbatimTextOutput("ANOVA1"),
                                                    plotOutput("assumptions1")))
                             ))),
                           
                           tabPanel(title = "Information about regression models",
                                    p("The tab of Modelling llvmi is a simple linear model that
                                      uses llvmi as a response, the plot below are the diagnoses plots"),
                                    br(),
                                    p("In the tab of Regression modelling you can choose the model,
                                      covariates and the response as you like, however the error message
                                      will produced once the response is not appropriate 
                                      for the regression model choosen,the plot below are the diagnoses plots")
                                    
                                    )
                             
                             
                             
                           ))


server<- function(input,output){
  output$sum<- renderPrint({
    #question1 summary
    summary(lvh[,input$var])
    
    
  })
  ###question 2 histogram
  
  output$thehist<- renderPlot({
    histo <- ggplot(lvh, aes(get(input$var), get(input$var0))) +
      geom_jitter() +
      geom_smooth(method = "loess", se=F)+
      xlab(input$var)+ylab(input$var0)
    ggMarginal(histo, type = "histogram", bins = input$bins)
    
    
  })
  
  ##question 3,4 
  output$vy<- renderUI({
    selectInput("vary","Select the second (Y) variable",choices = c("llvmi","sbp"))
  })
  output$vx<- renderUI({
    selectInput("varx","Select the first (X) variable",choices = c("sex","card","dgp","surv"))
  })
  
  
  
  ####boxplots
  output$thebox<- renderPlot({
    lvh$sex <- lvh$sex + 1
    lvh$card <- lvh$card + 1
    lvh$dgp <- lvh$dgp + 1
    #attach(get(lvh))
    #boxplot(data=lvh,x=get(as.character((input$varx))), y=get(input$vary),environment=environment())
    ggplot(lvh,aes(y=get(input$vary),x=as.factor(get((input$varx)))))+geom_boxplot()+
      xlab(input$varx)+ylab(input$vary)
    
    
  })
  
  ###density plot
  output$pairs<- renderPlot({
    lvh$sex <- lvh$sex + 1
    lvh$card <- lvh$card + 1
    lvh$dgp <- lvh$dgp + 1
    
    density <- ggplot(lvh, aes(get(input$vary)))+
      xlab(input$vary)
    density + geom_density(aes(fill=factor(get(input$varx))),alpha=0.8) +
      labs(title="Density Plot")+
      guides(fill=guide_legend(title= input$varx))
    
  })
  
  #question 5
  ##modelling llvmi
  output$cov<- renderUI({
    checkboxGroupInput("covariate","Select covariate variable(s)",choices = c("sex","card","dgp","surv","sbp"),"sex")
  })

  vari <- reactive({
    paste(input$covariate, collapse="+")
  })
  modelformula <- reactive({
    as.formula(sprintf('%s~%s', colnames(lvh)[1], vari()))
  })
  model <- reactive({
    lm(modelformula(), data=lvh)
  })  
  ###errors changing 
  noerror<- reactive({ validate(
    need(input$covariate != "", "")
    
  )})
  
  ##sumamry
  output$sum1 <- renderPrint({
    noerror()
    summary(model())})
  ###assumptions
  output$ANOVA<- renderPrint({
    noerror()
    anova(model())})
  
  output$assumptions <- renderPlot({
    noerror()
    par(mfrow=c(2,2))
    
    plot(model())
  })
  #######################################question 6
  output$response<- renderUI({
    selectInput("response","1.Select the response",choices = c("llvmi","sex","card","dgp","surv","sbp"),"llvmi")
  })
  
  output$cov1<- renderUI({
    checkboxGroupInput("covariate1","2.Select the covariate(s)",choices = c("llvmi","sex","card","dgp","surv","sbp"),"sex")
  })
  
  output$reg<- renderUI({
    selectInput("regmodel","3.Select the model",choices = c("Normal regression",
                                                            "logistic regression","Quasipoisson regression"))
  })
  ##########question 6 inputing 
  vari1 <- reactive({
    paste(input$covariate1, collapse="+")
  })
  
  formula <- reactive({
    as.formula(sprintf('%s~%s', input$response, vari1()))
  })
  ###normal 
  normodel <- reactive({
    glm(formula(), data=lvh,family = "gaussian")
  })
  
  #logistic
  logmodel <- reactive({
    glm(formula(), data=lvh,family = binomial(link="logit"))
  })
  
  ###quasipoisson
  quamodel <- reactive({
    glm(formula(), data=lvh,family = "quasipoisson")
  })
  
  #####errors changing 
  nonerror<- reactive({ validate(
    need(input$covariate1 != "", "")
    
  )})
  
  
  
  
  ################################################################inputing
  ##########summaries
  output$sum2<- renderPrint({
    nonerror()
    
    ##poisson summary
    if(input$regmodel== "Normal regression"){
      validate(
        need((input$response %in% c("llvmi", "sbp")), "Please select a suitable response")
      )
      summary(normodel())
    }
    ##logistic summary 
    else if(input$regmodel== "logistic regression"){
      validate(
        need((input$response %in% c("sex", "card", "dgp", "surv")), "Please select a suitable response")
      )      
      summary(logmodel())
      
    }
    ######Quasipoisson 
    else if(input$regmodel== "Quasipoisson regression"){
      validate(
        need((input$response %in% c("llvmi", "sbp")), "Please select a suitable response")
      )
      summary(quamodel())
      
    }
  })
  #################assumption plots 
  output$assumptions1<- renderPlot({
    nonerror()
    
    par(mfrow=c(2,2))
    ##normal assumptions plot
    if(input$regmodel== "Normal regression"){
      validate(
        need((input$response %in% c("llvmi", "sbp")), "")
      )
      plot(normodel())
    }
    ##logistic assumptions1 plot 
    else if(input$regmodel== "logistic regression"){
      validate(
        need((input$response %in% c("sex", "card", "dgp", "surv")), "")
      )
      plot(logmodel())
      
    }
    ######Quasipoisson assumptions plot
    else{
      validate(
        need((input$response %in% c("llvmi", "sbp")), "")
      )
      plot(quamodel())
      
    }
  })
}
shinyApp(ui,server)
