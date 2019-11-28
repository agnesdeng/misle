# misle
Multiple imputation through statistical learning

The R package `misle` is built using TensorFlow™, which enables fast numerical computation and thus provides a solution for large-scale multiple imputation.

`misle` is still at the early stage of development so lots of work have to be done before it is officially released. 

## Current version 
- **multiple imputation by variational autoencoders**
- **multiple imputation by denoising autoencoders(with dropout)**

## Expected to be done in 2020
- **implement multiple imputation by XGboost**

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
