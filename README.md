[![](https://img.shields.io/badge/Made%20With-R-9cf)](https://github.com/agnesdeng/misle)
[![](https://img.shields.io/badge/Version-1.0.0-brightgreen)](https://github.com/agnesdeng/misle)
[![](https://img.shields.io/badge/Lifecycle-Experimental-ff69b4)](https://github.com/agnesdeng/misle)
# misle
Multiple imputation through statistical learning

The R package `misle` is built using TensorFlow™, which enables fast numerical computation and thus provides a solution for large-scale multiple imputation.

`misle` is still at the early stage of development so lots of work have to be done before it is officially released. 

## Current version

- **multiple imputation by XGboost**

## Under development 
- **multiple imputation by variational autoencoders**
- **multiple imputation by denoising autoencoders(with dropout)**


## Install `misle` (It may have compatibility issues with python version/tensorflow version) 

Our package  `misle`  is built using `tensorflow`.  If P package  `tensorflow`  has been properly installed, users can directly install the newest version of  `misle`  from GitHub.

``` r
devtools::install_github("agnesdeng/misle")
library(misle)
```

If  `tensorflow` has not been installed, we recommend to use virtual environment to install it.

``` r
library(reticulate)

#By default, python package tensorflow would be installed in the virtual environment named 'r-reticulate' 
virtualenv_install(packages = c("tensorflow==1.14.0"))

#Install tensorflow R package
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(method="virtualenv",version="1.14.0",envname = "r-reticulate")

#Install misle 
devtools::install_github("agnesdeng/misle")
library(misle)

```

## Install `mixgb` (Highly recommend)
If users only want to use multiple imputation through XGBoost, please install this simplified R package `mixgb` instead.
```r
devtools::install_github("agnesdeng/mixgb")
library(mixgb)
```
## Example: multiple imputation through Denoising autoencoder
```r
#a data frame (consists of numeric/binary/multicalss variables) with NAs
withNA.df=createNA(adult,p=0.3)

#create a variational autoencoder imputer with your choice of settings or leave it as default
MIDAE=Midae$new(withNA.df,iteration=20,input_drop=0.2,hidden_drop=0.3,n_h=4L)

#training
MIDAE$train()

#impute m datasets
imputed.data=MIDAE$impute(m = 5)

#the 2nd imputed dataset
imputed.data[[2]]
```
## Example: multiple imputation through Variational autoencoder
```r
#a data frame (consists of  numeric/binary/multicalss variables) with NAs
withNA.df=createNA(adult,p=0.3)

#create a variational autoencoder imputer with your choice of settings or leave it as default
MIVAE=Mivae$new(withNA.df,iteration=20,input_drop=0.2,hidden_drop=0.3,n_h=4L)

#training
MIVAE$train()

#impute m datasets
imputed.data=MIVAE$impute(m = 5)

#the 2nd imputed dataset
imputed.data[[2]]
```

## Example: multiple imputation through XGBoost

We first load the NHANES dataset from the R package "hexbin".
``` r
library(hexbin)
data("NHANES")
```

Create 30% MCAR missing data.
``` r
#a dataframe (consists of numeric/binary/multicalss variables) with NAs
withNA.df<-createNA(NHANES,p=0.3)
```

Create an Mixgb imputer with your choice of settings or leave it as default.

Note that users do not need to convert the data frame into dgCMatrix or one-hot coding themselves. Ths imputer will convert it automatically for you. The type of variables should be one of the following: numeric, integer, or factor (binary/multiclass).
``` r
MIXGB<-Mixgb$new(withNA.df,pmm.type="auto",pmm.k = 5)
```

Use this imputer to obtain m imputed datasets.
``` r
mixgb.data<-MIXGB$impute(m=5)
``` 

## Example: impute new unseen data
First we can split a dataset as training data and test data.
``` r
set.seed(2021)
n=nrow(iris)
idx=sample(1:n, size = round(0.7*n), replace=FALSE)

train.df=iris[idx,]
test.df=iris[-idx,]
```

Since the original data doesn't have any missing value, we create some.
``` r
trainNA.df=createNA(train.df,p=0.3)
testNA.df=createNA(test.df,p=0.3)
```

We can use the training data (with missing values) to obtain m imputed datasets. Imputed datasets, the models used in training processes and some parameters are saved in the object `mixgb.obj`.

``` r
MIXGB=Mixgb.train$new(trainNA.df)
mixgb.obj=MIXGB$impute(m=5)
```
We can now use this object to impute new unseen data by using the function `impute.new( )`.  If PMM is applied, predicted values of missing entries in the new dataset are matched with training data by default. Users can choose to match with the new dataset instead by setting `pmm.new = TRUE`.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df)
test.impute
```

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE)
test.impute
```
Users can also set the number of donors for PMM when impute the new dataset. If  `pmm.k` is not set here, it will use the saved parameter value from the training object  `mixgb.obj`.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE, pmm.k=3)
test.impute
```

Similarly, users can set the number of imputed datasets `m`.  Note that this value has to be smaller than the one set in the training object. If it is not specified, it will use the same `m` value as the training object.

``` r
test.impute=impute.new(object = mixgb.obj, newdata = testNA.df, pmm.new = TRUE, m=4)
test.impute
```




## Expected to be done in 2021-2022


- **Simulation studies**


   to show whether multiple imputation using statistical learning (machine learning) techniques will lead to statistical valid inference. 

- **Visual diagnostics**


   includes plotting functions for users to check whether the imputed values are sensible


## Reference
JJ Allaire and Yuan Tang (2019). tensorflow: R Interface to 'TensorFlow'. R package version 2.0.0. https://github.com/rstudio/tensorflow

Tianqi Chen, Tong He, Michael Benesty, Vadim Khotilovich, Yuan Tang, Hyunsu Cho, Kailong Chen,Rory Mitchell, Ignacio Cano, Tianyi Zhou, Mu Li,Junyuan Xie, Min Lin, Yifeng Geng and Yutian Li (2019). xgboost: Extreme Gradient Boosting. R package version 0.90.0.2. https://CRAN.R-project.org/package=xgboost

JJ Allaire and François Chollet (2019). keras: R Interface to 'Keras'. R package version 2.2.4.1.9001. https://keras.rstudio.com

Rubin, D. B. (1987). Multiple imputation for nonresponse in surveys (1. print. ed.). New York [u.a.]: Wiley.

Vincent, P., Larochelle, H., Bengio, Y., \& Manzagol, P. (Jul 5, 2008). Extracting and composing robust features with denoising autoencoders. Paper presented at the 1096-1103. doi:10.1145/1390156.1390294 Retrieved from http://dl.acm.org/citation.cfm?id=1390294

Gal, Y., \& Ghahramani, Z. (2015). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. Retrieved from https://arxiv.org/abs/1506.02142

Gal, Y., Hron, J., & Kendall, A. (2017). Concrete Dropout. NIPS.

Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? NIPS.
 
Kingma, D. P., \& Welling, M. (2013). Auto-encoding variational bayes. Retrieved from https://arxiv.org/abs/1312.6114

Alex Stenlake & Ranjit Lall. Python Package MIDAS: Multiple Imputation with Denoising Autoencoders https://github.com/Oracen/MIDAS
