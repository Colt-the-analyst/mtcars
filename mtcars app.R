# Load necessary libraries
library(shiny)      # For building interactive web apps
library(glue)       # For dynamically creating Stan code with user inputs
library(janitor)    # For cleaning and managing datasets
library(tidyverse)  # For data manipulation and visualization
library(rstan)      # For Bayesian inference using Stan
library(bslib)      # For Bootstrap theming (improves UI design)

# Load the mtcars dataset
# This dataset contains information on 32 cars, including fuel efficiency (mpg)
# and engine characteristics.
data(mtcars)

# Prepare the data to be used in the Bayesian model
stan_data <- list(
  N = nrow(mtcars),  # Number of observations (cars)
  mpg = mtcars$mpg,  # Miles per gallon (dependent variable)
  wt = mtcars$wt,    # Weight of car (independent variable)
  hp = mtcars$hp,    # Horsepower (independent variable)
  disp = mtcars$disp, # Engine displacement (independent variable)
  drat = mtcars$drat  # Rear axle ratio (independent variable)
)

################################################################################
#                                   UI DESIGN                                   
################################################################################

# The User Interface (UI) defines the structure and appearance of the app.
ui <- fluidPage(
  
  # Apply a modern Bootstrap theme for styling
  theme = bs_theme(
    bootswatch = "cosmo",   # Light theme with clean styling
    primary = "#2c3e50",    # Dark blue theme color
    secondary = "#18bc9c",  # Teal accents
    base_font = font_google("Poppins")  # Google Font for a modern look
  ),
  
  # Main Title centered at the top of the page
  div(
    class = "text-center",
    h1("Bayesian Regression on mtcars Dataset", style = "color:#2c3e50; margin-top: 20px; font-weight: bold;")
  ),
  
  # Introduction section with explanations about the app and dataset
  fluidRow(
    column(12, 
           div(
             h3("About the Data", style = "color: #d4ac0d;"),
             p("This application performs Bayesian linear regression on the `mtcars` dataset, 
                which contains fuel efficiency (`mpg`) and various car attributes such as weight, horsepower, 
                engine displacement, and rear axle ratio. The goal is to model the relationship between 
                `mpg` (fuel efficiency) and these features using Bayesian inference."),
             
             h3("How to Use This App", style = "color: #d4ac0d;"),
             p("1. Adjust the prior hyperparameters using the input fields below."),
             p("2. Click 'Run Bayesian Model' to fit the model."),
             p("3. View the summary of the estimated parameters and the posterior distributions."),
             
             h3("Bayesian Modeling Process", style = "color: #d4ac0d;"),
             p("We use a Bayesian linear regression model where `mpg` is modeled as a function of car features. 
                The priors for the regression coefficients (β) are set as normal distributions, and the likelihood 
                is assumed to be Gaussian."),
             p("After fitting the model with Markov Chain Monte Carlo (MCMC) sampling, the app displays 
                the posterior distributions of the regression coefficients, which represent our updated 
                beliefs about the parameters given the data.")
           )
    )
  ),
  
  # Sidebar layout for user inputs and outputs
  sidebarLayout(
    sidebarPanel(
      h3("Set Prior Hyperparameters", style = "color:#2c3e50;"),
      p("Adjust the prior mean and standard deviation for each regression coefficient before running the model."),
      
      # Inputs for user-defined prior means and standard deviations
      numericInput("beta_0_mean", "Intercept Mean (β₀)", value = 20),
      numericInput("beta_0_sd", "Intercept SD", value = 10),
      numericInput("beta_wt_mean", "Weight Mean (β_wt)", value = 0),
      numericInput("beta_wt_sd", "Weight SD", value = 10),
      numericInput("beta_hp_mean", "Horsepower Mean (β_hp)", value = 0),
      numericInput("beta_hp_sd", "Horsepower SD", value = 10),
      numericInput("beta_disp_mean", "Displacement Mean (β_disp)", value = 0),
      numericInput("beta_disp_sd", "Displacement SD", value = 10),
      numericInput("beta_drat_mean", "Rear Axle Ratio Mean (β_drat)", value = 0),
      numericInput("beta_drat_sd", "Rear Axle Ratio SD", value = 10),
      
      # Button to run the Bayesian model
      actionButton("run_model", "Run Bayesian Model", class = "btn btn-primary btn-lg")
    ),
    
    mainPanel(
      h3("Model Summary", style = "color:#d4ac0d;"),
      p("This section displays the estimated posterior distributions for the model parameters, including the mean and standard deviation. NOTE THAT THIS MAY TAKE SOME TIME TO RUN."),
      verbatimTextOutput("model_summary"),
      
      h3("Posterior Distributions", style = "color:#d4ac0d;"),
      p("The graphs below show the posterior distributions of the regression coefficients. Each plot represents our updated belief about a parameter after incorporating the data."),
      plotOutput("posterior_plot")
    )
  )
)

################################################################################
#                                SERVER FUNCTION                                
################################################################################

server <- function(input, output, session) {
  observeEvent(input$run_model, {
    # Generate the Stan model using user-defined priors
    stan_code <- glue(
      "
    data {{
      int<lower=0> N;
      vector[N] mpg;
      vector[N] wt;
      vector[N] hp;
      vector[N] disp;
      vector[N] drat;
    }}

    parameters {{
      real beta_0;
      real beta_wt;
      real beta_hp;
      real beta_disp;
      real beta_drat;
      real<lower=0> sigma;
    }}

    model {{
      // Priors using user-defined hyperparameters
      beta_0 ~ normal({input$beta_0_mean}, {input$beta_0_sd});
      beta_wt ~ normal({input$beta_wt_mean}, {input$beta_wt_sd});
      beta_hp ~ normal({input$beta_hp_mean}, {input$beta_hp_sd});
      beta_disp ~ normal({input$beta_disp_mean}, {input$beta_disp_sd});
      beta_drat ~ normal({input$beta_drat_mean}, {input$beta_drat_sd});

      sigma ~ cauchy(0, 5);

      // Likelihood
      mpg ~ normal(beta_0 + beta_wt * wt + beta_hp * hp + beta_disp * disp + beta_drat * drat, sigma);
    }}
    "
    )
    
    # Run Bayesian regression using Stan
    fit <- stan(
      model_code = stan_code,
      data = stan_data,
      iter = 2000,
      chains = 4
    )
    
    # Output model summary
    output$model_summary <- renderPrint({
      print(fit,
            pars = c(
              "beta_0",
              "beta_wt",
              "beta_hp",
              "beta_disp",
              "beta_drat",
              "sigma"
            ))
    })
    
    # Extract posterior samples
    posterior_samples <- extract(fit, pars = c("beta_wt", "beta_hp", "beta_disp", "beta_drat"))
    posterior_df <- as.data.frame(posterior_samples)
    
    # Convert to tidy format for ggplot
    posterior_tidy <- posterior_df %>%
      pivot_longer(cols = everything(),
                   names_to = "Parameter",
                   values_to = "Value")
    
    # Plot posterior distributions
    output$posterior_plot <- renderPlot({
      ggplot(posterior_tidy, aes(x = Value)) +
        geom_density(fill = "#f1c40f", alpha = 0.8) +
        facet_wrap( ~ Parameter, scales = "free") +
        labs(title = "Posterior Distributions of Regression Coefficients", x = "Parameter Value", y = "Density") +
        theme_minimal()
    })
  })
}

# Run the Shiny app
shinyApp(ui, server)
