# Load necessary libraries
library(lme4)
library(MASS)
library(nnet)
library(rsq)
library(sigmoid)
library(haven)

# Load necessary libraries
library(Matrix)
library(lme4)
library(MASS)
library(sigmoid)
library(haven)
library(mosaic)

my_data <- read_dta("https://stats.idre.ucla.edu/stat/examples/imm/imm10.dta")
View(my_data )

my_data$sex<-as.factor(my_data$sex)
contrasts(my_data$sex)

##Multilevel logistic models
#We use the glmer() function which specify that  a logistic regression model, use the family=binomial(link="logit") option.
# # Run Binomial logistic model with random intercept
model1 <- glmer(sex ~ homework + (1|schid), data=my_data,
               family=binomial(link="logit"))
summary(model1)
hist(model1)


sigmoid(-0.14868)
sigmoid(0.05828)

exp(-0.14868)/(1+exp(-0.14868))
# Run logistic model with random intercept and slope
model2 <- glmer(sex ~ homework + (1 + homework | schid),  data=my_data,
               family=binomial(link="logit"))
summary(model2)

sigmoid(-0.17722 )
sigmoid(0.08423)
anova(model1,model2)

#We use the glmer() function which specify that  a poisson regression model, use the family=poisson(link="logit") option.
# # Run Poisson log link model with random intercept
PoissonModel1 <- glmer(homework ~ math + (1|schid), data=my_data,
                family=poisson(link="log"))

summary(PoissonModel1)
exp(-0.872639)
exp(0.027857)
#The expected log count for each unit increase/decrease (depending on the sign of the coefficient) in [outcome variable] given [predictor variable] is [coefficient].


# Run poisson log link model with random intercept and slope
PoissonModel2 <- glmer(homework ~ 1+math+ (1 + math | schid),  data=my_data,
                family=poisson(link="log"))
summary(PoissonModel2)
exp(-0.093660)
exp(0.009244)
#Inverse of log link function


#Model comparsion
anova(PoissonModel1,PoissonModel2)

