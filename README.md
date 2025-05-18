
### Discussing Different Likelihoods and Link Functions, and Their Influence on the Interpretation of Results in Generalized Linear Multilevel Model

## Overview
Multilevel Generalized Linear Models (MGLMs) extend the capabilities of Multilevel Linear Models (MLMs). While MLMs are specifically tailored for continuous outcomes, MGLMs are adept at handling non-continuous variables such as binary, count, or categorical outcomes.

In essence, MLMs are well-suited for analyzing smoothly changing numeric data like heights or scores but MGLMs are better equipped to handle categorical, count, or binary data such as yes/no answers or color categories. By selecting the appropriate model based on the nature of the data, researchers can ensure more accurate and meaningful analyses.

In generalized linear models (GLMs), the relationship between the linear predictor and the mean of the response variable is established through the link function. The link function connects the linear predictor to the response variable's expected value, while the inverse link function performs the reverse transformation.

In our research, we are considering three discrete distributions(Binomial, Poisson, Negative-binomial) and their likelihoods and link functions on the interpretation of the result.For Poisson, binomial, and negative binomial distributions, common link functions include the log link for Poisson and negative binomial distributions, and the logit link for binomial distributions. 

## Multilevel data structures 
GLMs offer a versatile framework for analyzing hierarchical data structures with diverse outcome types, making them valuable in a range of research scenarios. For instance, in educational research, students (level 1) nested within schools (level 2) form a multilevel structure. Each school contains multiple students, and observations are not independent within schools. This structure requires modeling the variation at both the student and school levels, making multilevel modeling appropriate for understanding individual and group-level effects simultaneously.

## Data Description
The data containing, at the student level, information about math scores, socioeconomic status, sex, race, and other student characteristics. School level characteristics include mean socioeconomic status, urbanicity, teacher/student ratios, and other characteristics. The data can be learned from ("https://stats.idre.ucla.edu/stat/examples/imm/imm10.dta"). Description of some variables are given below:

|Variable|Description and codes |  
|:---------|---------:|
|  schid      |  School id   |       
|  sex   |  1=Male, 2= Female |
|  math   |  information about math scores |
|  homework   |  time spent on math homework each week  |
|  white  | race: 1= white, 2= non-white |
|  ses  | socioeconomic status |


```{r}
# Load necessary libraries
library(lme4)
library(sigmoid)
library(haven)
library(mosaic)
library(DT)
```

```{r}
# Loading the datasets
my_data <- read_dta("https://stats.idre.ucla.edu/stat/examples/imm/imm10.dta")
my_data <- my_data[,c("schid","homework","sex", "math")]

```

```{r}
# Use as.factor() R function to converts sex variable column from numeric to factor.
my_data$sex<-as.factor(my_data$sex)
# The main function datatable() is used from Library 'DT' which allows easy viewing of data. It creates an HTML widget to display R data objects with DataTables.
DT::datatable(data = my_data, rownames = FALSE,class = 'cell-border stripe', colnames = c('schid', 'homework', 'sex', 'math'))
```
```{r}
# Descriptive statistics of the datasets
summary(my_data)
```
## Data Visualization
Here, bargraph is used to display COUNTS for the number of observations in our datafile within certain categories. We view the number of students who are male or female in each school, or number of hours spend on homework by the type of student(male/female) and attaining math score based on time spent on homework each week for each school.
```{r}
bargraph(~sex,  groups=schid, data=my_data, main = "Sex grouped by School id")
bargraph(~sex,  groups=homework, data=my_data, main = "Sex grouped by homework for each School id")
bargraph(~math,  groups=homework, data=my_data, main = " Homework grouped by Mathscore for each school id")
```

## Multilevel Generalized linear model for Binomial distribution

### Model description

**Model 1: Random Intercept model with fixed slope**

We have specified that the intercept varies by group (which is, in this case, the school). Here, we include a model for the random intercepts, but without a random slope.
$$
\log(p_{ij}) = \beta_{00} + \beta_{10} \times \mathrm{homework}_{ij} + u_{0j} + \epsilon_{ij}
$$
                   
**Model 2: Random Intercept model and  random slope**

For the random intercept and random slope model, we have specified that the intercept varies by group and also includes random slope for parameter $\beta_{1}$.

$$
\log(p_{ij}) = \beta_{00} + \beta_{10} \times \text{homework}_{ij} + u_{0j} + u_{1j} \times \text{homework}_{ij} + \varepsilon_{ij}
$$

Where,
$p_{ij}$ is the probability of observing the positive outcome for the  $i^{th}$ student in the $j{th}$ school.

$\beta_{00}$ and $\beta_{10}$ are the fixed intercept and fixed slope respectively

$\text{homework}_{ij}$ denotes the predictor variable for observation $i$ in group $j$

$u_{0j}$ and $u_{1j}$ are random intercepts and random slopes,grouped by school id $j$

$\varepsilon_{ij}$ is the error term for observation $i$ in group $j$
              
### Likelihood function

Likelihood is the probability of observing the given data considering the model parameter.
For the $i^{th}$ student in the $j{th}$ school, the likelihood of observing their sex (binary outcome) is modeled by using a Bernoulli distribution. 

$$
f(y_{ij} \,|\, \beta_{0j}, \beta_{1j}) = p_{ij}^{y_{ij}} \times (1 - p_{ij})^{1 - y_{ij}}
$$
Combining the individual likelihoods across all students within a school and considering the hierarchical structure the likelihood function can be represented as:

$$
L(\beta_0, \beta_1) = \prod_{j=1}^{J} \left[ \prod_{i=1}^{n_j} p_{ij}^{y_{ij}} \times (1 - p_{ij})^{1 - y_{ij}} \right]
$$
Here, $j$ is the total number of schools, $n_j$  is the number of students in the $j{th}$  school and $y_{ij}$ is the binary outcome for the $i^{th}$ student in the $j{th}$ school.

### Link functions

**Logit Link Function**: The logit link function is commonly used for binary outcomes in logistic regressions which connects the linear predictor to the mean of the distribution.For the binomial model, the likelihood function is binomial (n, p), where 'n' is the number of trials and 'p' is the probability of success i.e, 
$$y \sim \text{Binomial}(n, p)$$

The odds of success are defined as the ratio of the probability of success over the probability of failure. If p is the probability of an event, then, $Odds = \frac{p}{1 - p}$. The natural logarithm of the odds known as the “log-odds” or “logit” which is the link function for our binary outcomes.
$$
\text{logit}(p) = \log\left(\frac{p}{1-p}\right) = \eta
$$
and $$\eta = \beta_0 + \beta_1 X_i$$ is the linear predictor. To convert log-odds to a probability, use the inverse logit function which is called sigmoid function, maps any real valued number to a value between 0 and 1. 

$$
\frac{e^x}{1+e^x}=\frac{1}{1 + e^{-x}}
$$

### Run Binomial logistic model with random intercept

```{r}
binom_model1 <- glmer(sex ~ homework + (1|schid), data=my_data,
               family=binomial(link="logit"))
summary(binom_model1)
```

```{r}
#Applying the sigmoid function to the predictor with logit link ensures the predicted  probabilities between 0 and 1.
sigmoid(-0.14868)
sigmoid(0.05828)
```

#### Interpretation

**Fixed Effects**

**Intercept**: The estimated probability of event “sex” being true approximately 0.463.The p-value suggests that the effect is not statistically significant (p = 0.467). 

**Homework**: The estimated value for homework predictor is Sigmoid(0.05828)= 0.5145659
The p-value suggests that this effect is not statistically significant (p = 0.468). 

**Correlation((Intr))**: The correlation between the fixed intercept and the fixed effect of sex is approximately -0.795 which means the time spent on math homework each week is negatively correlated with sex.

**Random Effects**

There are random intercepts for each 'schid' group. The variance is estimated to be 0, which means there's no variability between groups in terms of intercepts.


### Run binomial logistic model with random intercept and slope
```{r}
binom_model2 <- glmer(sex ~ homework + (1 + homework | schid),  data=my_data,
               family=binomial(link="logit"))
summary(binom_model2)
```

```{r}
#Applying the sigmoid function to the predictor with logit link ensures the predicted  probabilities between 0 and 1.
sigmoid(-0.17722 )
sigmoid(0.08423)

```
#### Interpretation

**Fixed Effect**

**Intercept**: The estimated probability of event “sex” being true approximately 0.456.The p-value suggests that the effect is not statistically significant (p = 0.534). 

**Homework**: The estimated value for homework predictor is Sigmoid(0.08423)= 0.5210451.However, similar to the previous model, the p-value suggests that this effect is not statistically significant (p = 0.671). 

**Random Effects** 

There are random intercepts for each 'schid' group, with a small estimated variance (0.0003211) and standard deviation (0.01792). 
Additionally, 
There are random slopes for 'homework' within each 'schid' group. The estimated variance is 0.0032509, and the standard deviation is 0.05702. 

### Model comparison
```{r}
anova(binom_model1,binom_model2)
```
**Likelihood Ratio Test (Chisq)**

The chi-square statistic is 0.0108 with 2 degrees of freedom, comparing model2 to model1. 

The p-value for the chi-square test is 0.9946, indicating that the additional parameters in model2 do not significantly improve model fit. 

The likelihood ratio test suggests that the more complex model (model2) with additional parameters does not provide a significantly better fit than the simpler model (model1).


## Multilevel Generalized linear model for Poisson distribution

### Model description

**Model 1: Random Intercept model with fixed slope**

We have specified that the intercept varies by group (which is, in this case, the school). Here, we include a model for the random intercepts, but without a random slope.
$$
\log(p_{ij}) = \beta_{00} + \beta_{10} \times \mathrm{math}_{ij} + u_{0j} + \epsilon_{ij}
$$
**Model 2: Random Intercept model and  random slope**

For the random intercept and random slope model, we have specified that the intercept varies by group and also includes random slope for parameter $\beta_{1}$.

$$
\log(p_{ij}) = \beta_{00} + \beta_{10} \times \text{math}_{ij} + u_{0j} + u_{1j} \times \text{math}_{ij} + \varepsilon_{ij}
$$

Where,

$p_{ij}$ is the probability of observing the positive outcome for the  $i^{th}$ student in the $j{th}$ school.

$\beta_{00}$ and $\beta_{10}$ are coefficients

$\text{math}_{ij}$ denotes the predictor variable for observation $i$ in group $j$

$u_{0j}$ and $u_{1j}$ are random intercepts and random slopes for group $j$

$\varepsilon_{ij}$ is the error term for observation $i$ in group $j$

### Likelihood functions
The likelihood function is the joint probability of observed data viewed as a function of the parameters of a statistical model. For the $i^{th}$ student in the $j{th}$ school, the likelihood of observing number of hours spent in homework is modeled by using a Poisson distribution.

$$
f(\beta, \lambda_i) = \prod_{i=1}^{n_j} e^{-\lambda_{ij}}  \lambda_{ij}^k/ k!
$$
Combining the individual likelihoods across all students within a school and considering the hierarchical structure:

$$
L(\beta_0, \beta_1) = \prod_{j=1}^{J} \left[ \prod_{i=1}^{n_j} \frac{e^{-\lambda_{ij}} \lambda_{ij}^k}{k!} \right]
$$

Where,

$j$ is the total number of schools, $n_j$  is the number of students in the $j{th}$  school and $y_{ij}$ is the number of hours spent for homework for the $i^{th}$ student in the $j{th}$ school.

$f(\beta, \lambda_i)$ is the function with parameters $\beta$ and $\lambda_i$

$\lambda_{ij}$ represents the rate parameter for observation $i$ in group $j$

### Link functions
The link function for the Poisson distribution is the natural logarithm. Poisson GLM can be expressed as: 

   $$g(p) = \log(p)$$
Where, g(.) is the link function  and p is the expected value of the response variable. For Poisson regression three components of GLM , formulated like this.

$$
y_i \sim \text{Poisson}(\lambda_i)
$$

$$
\ln(\lambda_i) = \beta_0 + \beta_i x_i = \eta
$$
where, $\ln(\lambda_i)$ is link function and $\eta$ is the linear predictor for the covariates $x_{i}$. Inverse of the log link function is an exponential function, implies parameter for Poisson regression calculated by the linear predictor guaranteed to be positive.


$$
\lambda_i = \exp(\beta_0 + \beta_i x_i)
$$

### Run Poisson log link model with random intercept
```{r}
PoissonModel1 <- glmer(homework ~ math  + (1|schid), data=my_data,
                family=poisson(link="log"))
summary(PoissonModel1)
```

```{r}
#Inverse of log link function
exp(-0.872639)
```

#### Interpretation

**Fixed effects**

**Intercept**: On average, the expected number of hours spent on homework is about 41.7% of the reference when the math score is zero. The p-value suggests that the effect is statistically significant (p = 0.0017).

**Math**: For a one-unit increase in math scores, the expected number of hours spent on homework increases by approximately 2.8%.
A significant positive association between math scores and number of hours spent on homework, accounting for random effects associated with different schools.

**Random effects**:

**Intercept(schid)**: There is variability in the intercepts among schools, suggesting differences in the baseline number of hours spent on homework.


### Run poisson log link model with random intercept and slope
```{r}
PoissonModel2 <- glmer(homework ~ math+ (1 + math | schid),  data=my_data,
                family=poisson(link="log"))
summary(PoissonModel2)
```

```{r}
#Inverse of log link function
exp(-0.093660)
```
#### Interpretation:

**Fixed effects**:

**Intercept**: On average, the expected number of hours spent on homework is about 91.05% of the reference when the math score is zero. 

**Math**: For a one-unit increase in math scores, the expected log-odds for number of hours spent on homework increases by approximately 0.9%.

**Random effects**:

**Schid(Intercept)**: 4.001 indicates Variability among schools in the baseline homework hours.
Math(0.00194): Variability among schools in the effect of math scores on homework hours

### Model comparsion
```{r}
anova(PoissonModel1,PoissonModel2)
```
**Likelihood Ratio Test (Chisq)**

The chi-square statistic is 24.15 with 2 degrees of freedom, comparing model2 to model1. 

The p-value for the chi-square test is less than 0.05, indicating that the additional parameters in model2 significantly improve model fit. 

The likelihood ratio test suggests that the more complex model (model2) with additional parameters provide a significantly better fit than the simpler model (model1).

## Multilevel Generalized linear model for Negative binomial distribution
### Model description
Let random variable  Y be some count, $y \in \{0,1, 2,...\}$, that can be modeled by the Negative Binomial with mean parameter $\mu$ and reciprocal dispersion parameter $r$
$$
Y|μ,r \sim \text{NegBin}(μ,r)
$$
with unequal mean $$E(Y|μ,r)= μ$$ and variance $$Var(Y|μ,r)= μ+ μ^2/r$$

If the reciprocal dispersion parameter r is large then $Var(Y) ≈ E(Y)$ and $Y$ behaves as like as poisson count variable. For small r, $Var(y) > E(Y)$ and Y is overdispersed in comparison to a Poisson count variable with the same mean.

We can keep reciprocal dispersion parameter, $r>0$ and switch to more flexible Negative Binomial regression model simply changing a Poisson data model for a Negative Binomial data model


#### Likelihood functions

The negative binomial likelihood function can be written as (similar as poisson Likelihood function):

$$
L(\beta_0, \beta_1) = \prod_{j=1}^{J} \left[ \prod_{i=1}^{n_j} \frac{e^{-\lambda_{ij}}}{k!} \lambda_{ij}^k \right]
$$
Where,

$\boldsymbol{\beta}_0$ and $\boldsymbol{\beta}_1$ are the parameters of interest

$J$ is the number of groups
 
$n_j$ is the number of observations in group $j$

$\lambda_{ij}$ is the mean parameter for observation $i$ in group $j$

$k$ is the dispersion parameter


### Link functions
The link function in the negative binomial regression model is typically the log link, which ensures that the predicted values are positive.The negative binomial regression model with log link function can be expressed as:

$$
\log(\mu_i) = \mathbf{X}_i \boldsymbol{\beta}= \eta
$$

Where,

$\mu_i$ is the expected value of the response variable for observation $i$.

$\mathbf{X}_i$ is the predictor matrix for observation $i$.

$\boldsymbol{\beta}$ is the vector of coefficients.


The equation for the negative binomial model (similar to Poisson model) with log link function can be written as:

\[
\log(p_{ij}) = \beta_{00} + \beta_{10} \times \text{math}_{ij} + u_{0j} + \varepsilon_{ij}
\]

Where,

\( p_{ij} \) is the probability of occurrence for observation \( i \) in group \( j \)

\( \beta_{00} \) and \( \beta_{10} \) are the coefficients

\( \text{math}_{ij} \) represents the predictor variable for observation \( i \) in group \( j \) (in this case, \( \text{math}_{ij} \))

\( u_{0j} \) is the random effect for group \( j \)

\( \varepsilon_{ij} \) is the error term for observation \( i \) in group \( j \)

### Run negative binomial log link model with random intercept
```{r}
NegativeBinomialModel1 <- glmer.nb(homework ~ math+ (1|schid), data=my_data, family=NegativeBinomialModel(link="log"))
summary(NegativeBinomialModel1)
```

```{r}
#Inverse of log link function
exp(-0.872700)

```
### Interpretation

**Fixed effects**

**Intercept**: On average, the expected number of hours spent on homework is about 41.8% of the reference when the math score is zero. The p-value suggests that the effect is statistically significant (p = 0.0017).

**Math**: For a one-unit increase in math scores, the expected number of hours spent on homework increases by approximately 2.78%. A significant positive association between math scores and number of hours spent on homework, accounting for random effects associated with different schools.

**Random effects**

**Intercept(schid)**: There is variability in the intercepts among schools, suggesting differences in the baseline number of hours spent on homework.


### Run negative binomial log link model with random intercept and random slope
```{r}
NegativeBinomialModel2 <- glmer.nb(homework ~ math + (1 + homework | schid), 
                                  data = my_data, family=NegativeBinomialModel(link="log"))
summary(NegativeBinomialModel2)

```

```{r}
#Inverse of log link function
exp(-0.528148)

```
#### Interpretation

**Fixed effects**

**Intercept**: On average, the expected number of hours spent on homework is about 58.96% of the reference when the math score is zero.

**Math**: For a one-unit increase in math scores, the expected log-odds for number of hours spent on homework increases by approximately 0.365%.

**Random effects**

**Schid(Intercept)**: 4.386e-19 indicates Variability among schools in the baseline homework hours.

**Math(1.780e-01)**: Variability among schools in the effect of math scores on homework hours

### Model comparsion
```{r}
anova(NegativeBinomialModel1,NegativeBinomialModel2)
```
**Likelihood Ratio Test (Chisq)**

The chi-square statistic is 129.99 with 2 degrees of freedom, comparing model2 to model1. 

The p-value for the chi-square test is less than 0.05, indicating that the additional parameters in model2 significantly improve model fit. 

The likelihood ratio test suggests that the more complex model (model2) with additional parameters provide a significantly better fit than the simpler model (model1).

## Limutations and challenges
When dealing with discrete distributions, such as the binomial, Poisson, or negative binomial distributions, a few limitations and challenges come into play:

1. Interpretation of Results: The choice of likelihood and link functions can influence the interpretation of results. For example, using a binomial distribution with a logit link function may be appropriate for binary outcomes, but the interpretation of coefficients may differ compared to using a Poisson distribution with a log link function for count outcomes.

2. Computational Complexity: Estimating generalized linear multilevel models with discrete distributions can be computationally intensive.

3. glmer() estimates model parameters using Maximum Likelihood (ML) estimation. While ML estimation is widely used and computationally efficient, it comes with certain limitations.

## Conclusion
Throughout this tutorial, initially we explained the concept of generalized linear models used in multilevel approach. We checked over the influence of different likelihoods and link functions on the interpretation of the results for MGLMs discrete distribution such as binomial, Poisson and also negative binomial which is much like to poisson distribution for large reciprocal dispersion parameter. 
For this purpose, we fit two types of model including random intercept and random slope. Therefore, comparing these fitted-model, for the binomial distributions, model including random slope significantly not fit better. For Poisson and negative binomial  distribution, more complex model including random slope significantly fit better.

## Referrence
Finch, W. H., Bolin, J. E., & Kelley, K. (2019). Multilevel modeling using R. Chapman and Hall/CRC.

Gelman, A., & Hill, J. (2006). Data analysis using regression and multilevel/hierarchical models . Cambridge university press. Link [Ch.15] book

Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.

Nalborczyk, L., Batailler, C., Lœvenbruck, H., Vilain, A., & Bürkner, P. C. (2019). An introduction to Bayesian multilevel models using brms: A case study of gender effects on vowel variability in standard Indonesian. Journal of Speech, Language, and Hearing Research, 62(5), 1225-1242.

Raudenbush, S. W., & Bryk, A. S. (2002). Hierarchical Linear Models: Applications and Data Analysis Methods (2nd ed.). Sage Publications.

Snijders, T. A. B., & Bosker, R. J. (2012). Multilevel Analysis: An Introduction to Basic and Advanced Multilevel Modeling (2nd ed.). SAGE Publications.

https://www.dataquest.io/blog/tutorial-poisson-regression-in-r/

https://www.jstatsoft.org/article/view/v100i05

https://rpubs.com/rslbliss/r_mlm_ws

https://www.bayesrulesbook.com/chapter-13

https://www.bayesrulesbook.com/chapter-12









