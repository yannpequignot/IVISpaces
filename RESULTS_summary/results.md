# RESULTS: February 2, 2021
	

# Uncertainty score

For each input $x$ and ensemble $F$ of predictor the epistemic uncertainty can be measured using an estimate of

* the standard deviation (or equivalently the variance) of the predictions $\{ f(x) \mid f\in F\}$
* the differential entropy of the predictions $\{ f(x) \mid f\in F\}$



As a score for OOD discrimination, the variance performs better than the differential entropy.

For a normal $\mu \sim N(0,\sigma)$ we have $H(\mu)=\frac{1}{2} \log(2\pi e \sigma^2)$. So the entropy is strictly increasing function of the standard deviation $\sigma>0$. So using the variance amounts to assuming the predictions at a given $x$ follows a normal distribution (an assumption which is already made for Ensemble and MC dropout for the LPP on test).

Moreover for ever $\sigma>0$, the normal distribution $N(0,\sigma)$ has the largest possible entropy among distributions with standard deviation $\sigma$. 

Intuition, $\{ f(x) \mid f\in F\}$ does not seem to follow a normal distribution, and even more so when it has large variance.

![](VarVsEntropy_Exp2Small.png)

**We henceforth use only the variance of the predictions as a score of a uncertainty, or in fact the standard deviation**.


# OOD detection evaluation

OOD samples are genereated using a uniform distribution on the hyperrectangle based on the TRAINING inputs. 

Here we plot the distribution of the Euclidean distance to the nearest neighbour in the TRAINING inputs for 10K OOD samples and the test inputs.

## small UCI

### Exp1
![](NN_dist_to_train_smallUCI.png)

### Exp2 (10 different splits)
![](NN_dist_to_train_smallUCI_exp2.png)

## large UCI
![](NN_dist_to_train_largeUCI.png)




# EXP1

Except for Ensemble which minimizes the square loss, all the other methods use a Gaussian likelihood based on choice of a value of sigma_noise: the aleatoric uncertainty. Here sigma_noise is not learned, it fixed to a specific value that was inspired by Yarin Gal's grid search.


## RMSE

![](Exp1_LPP.png)

<!--
\begin{tabular}{llllllll}
\toprule
{} &         HMC &  MC dropout &    Ensemble &         MFVI &   FuNN-MFVI &     NN-HyVI &   FuNN-HyVI \\
dataset  &             &             &             &              &             &             &             \\
\midrule
boston   &   2.63±0.02 &  2.29±0.034 &  2.73±0.039 &    3.5±0.009 &  4.97±0.708 &  2.65±0.016 &  3.05±0.105 \\
concrete &  6.96±0.033 &  5.09±0.075 &  4.52±0.042 &  12.25±0.022 &  6.31±0.541 &  6.06±0.177 &   5.19±0.21 \\
energy   &  2.62±0.008 &  0.74±0.008 &  0.41±0.003 &   3.41±0.005 &  0.79±0.141 &  0.68±0.014 &  0.49±0.009 \\
wine     &    0.58±0.0 &   0.6±0.009 &  0.57±0.004 &    0.62±0.01 &   0.7±0.002 &  0.73±0.006 &   0.72±0.03 \\
yacht    &  3.84±0.022 &  1.06±0.043 &  0.86±0.035 &   3.89±0.028 &  2.25±0.184 &  1.28±0.044 &  1.07±0.089 \\
\bottomrule
\end{tabular}
-->
## LPP
![](EXP1_RMSE.png)

<!--
\begin{tabular}{llllllll}
\toprule
{} &           HMC &    MC dropout &      Ensemble &          MFVI &     FuNN-MFVI &       NN-HyVI &     FuNN-HyVI \\
dataset  &               &               &               &               &               &               &               \\
\midrule
boston   &  -4.1±0.00018 &  -4.1±0.00014 &   -3.2±0.2008 &  -4.1±0.00016 &  -4.1±0.01011 &  -4.1±0.00018 &  -4.1±0.00086 \\
concrete &   -5.3±0.0001 &    -5.2±7e-05 &    -3.6±0.159 &  -5.3±0.00017 &  -5.2±0.00075 &  -5.2±0.00024 &  -5.2±0.00016 \\
energy   &   -3.6±0.0002 &  -3.6±0.00012 &  -1.6±0.47242 &  -3.6±0.00018 &  -3.6±0.00056 &  -3.6±0.00017 &    -3.6±2e-05 \\
wine     &  -0.9±0.00074 &  -1.0±0.02606 &  -1.5±0.06858 &  -1.1±0.03673 &  -1.5±0.00982 &   -1.5±0.0522 &  -1.6±0.13075 \\
yacht    &  -4.1±0.00045 &   -4.0±0.0001 &  -0.5±0.08407 &  -4.1±0.00062 &  -4.0±0.00285 &    -4.0±7e-05 &    -4.0±8e-05 \\
\bottomrule
\end{tabular}
-->

## logs

![](Exp1_logs.png)

## Epistemic predictive variance distributions


### histogram
![](Exp1_EpistVarDist_Exp2-1.pdf)

scaled on variance range $[0,3]$.

![](Exp1_EpistVarDist_Exp2-scaled_0to3.pdf)
### empirical cumulative distribution
![](Exp1_EpistVarECDF_Exp2-1.pdf)


## ROC curves and AUC

![](Exp1_ROC_curves_VAR_MinMax.pdf)

## Concrete detail

on $[0,4]$
![](Exp1_EpistVarECDF_Exp2-concrete.pdf)
on $[0,15]$
![](Exp1_EpistVarECDF_Exp2-concrete_to15.pdf)![](Exp1_ROC_curves_concrete.pdf)



## Entropy of ensemble

For the ensemble method: 10 models, others 1K models. 
Parameter space entropy not available for MC dropoout.

### Parameter space

![](Exp1_Param_Entropy.png)

<!--
\begin{tabular}{llllllll}
\toprule
{} &        HMC & MC dropout &   Ensemble &        MFVI &      FuNN-MFVI &      NN-HyVI &      FuNN-HyVI \\
{} &  parameter &  parameter &  parameter &   parameter &      parameter &    parameter &      parameter \\
\midrule
boston   &  970.0±1.0 &    nan±nan &  332.0±5.0 &   567.0±1.0 &  -1117.0±289.0 &  1555.0±25.0 &     350.0±27.0 \\
concrete &  635.0±1.0 &    nan±nan &  424.0±2.0 &   353.0±1.0 &    -697.0±60.0 &   1078.0±2.0 &      431.0±9.0 \\
energy   &  628.0±1.0 &    nan±nan &  193.0±4.0 &   348.0±1.0 &   -465.0±126.0 &  1098.0±12.0 &     334.0±52.0 \\
wine     &  615.0±4.0 &    nan±nan &  812.0±6.0 &  247.0±30.0 &  -1035.0±183.0 &  1455.0±59.0 &  -1596.0±165.0 \\
yacht    &  502.0±0.0 &    nan±nan &  152.0±4.0 &   263.0±0.0 &    -589.0±85.0 &    772.0±8.0 &     145.0±78.0 \\
\bottomrule
\end{tabular}
-->

### Predictor space
![](Exp1_Predictor_Entropy.png)

<!--
\begin{tabular}{llllllll}
\toprule
{} &         HMC &   MC dropout &     Ensemble &         MFVI &   FuNN-MFVI &      NN-HyVI &   FuNN-HyVI \\
{} &   predictor &    predictor &    predictor &    predictor &   predictor &    predictor &   predictor \\
\midrule
boston   &   -52.0±1.0 &   -491.0±7.0 &   -171.0±6.0 &   -317.0±0.0 &   122.0±3.0 &   -398.0±6.0 &  294.0±10.0 \\
concrete &  -107.0±1.0 &  -492.0±19.0 &  -211.0±16.0 &   -316.0±0.0 &    76.0±2.0 &   -423.0±7.0 &   273.0±2.0 \\
energy   &  -224.0±2.0 &  -506.0±67.0 &  -346.0±13.0 &   -587.0±0.0 &  -10.0±17.0 &  -301.0±20.0 &  160.0±10.0 \\
wine     &   -25.0±1.0 &  -234.0±33.0 &      7.0±3.0 &  -422.0±10.0 &  128.0±34.0 &  -285.0±24.0 &  254.0±23.0 \\
yacht    &  -244.0±1.0 &   -844.0±7.0 &   -574.0±8.0 &   -492.0±0.0 &  -83.0±10.0 &   -667.0±7.0 &  113.0±15.0 \\
\bottomrule
\end{tabular}
-->

# EXP2 small

## RMSE

![](Exp2_small_RMSE.png)

<!--
\begin{tabular}{lllllll}
\toprule
{} &  MC dropout &    Ensemble &        MFVI &   FuNN-MFVI &     NN-HyVI &   FuNN-HyVI \\
dataset  &             &             &             &             &             &             \\
\midrule
boston   &  3.445±0.34 &  3.393±0.34 &  3.808±0.25 &  4.263±0.48 &   3.77±0.48 &  3.733±0.23 \\
concrete &  4.707±0.18 &  4.126±0.23 &  5.108±0.15 &  5.644±0.24 &   4.72±0.16 &  4.749±0.27 \\
energy   &  0.566±0.02 &  0.411±0.02 &  0.463±0.02 &  0.573±0.03 &  0.437±0.04 &  0.457±0.04 \\
wine     &   0.63±0.01 &  0.641±0.02 &   0.65±0.01 &  0.674±0.01 &  0.736±0.01 &  0.696±0.02 \\
yacht    &  0.737±0.07 &   0.68±0.07 &  1.153±0.07 &  1.708±0.17 &  0.599±0.07 &  0.911±0.09 \\
\bottomrule
\end{tabular}
-->
## LPP
![](Exp2_small_LPP.png)

<!--
\begin{tabular}{lllllll}
\toprule
{} &   MC dropout &     Ensemble &         MFVI &    FuNN-MFVI &      NN-HyVI &     FuNN-HyVI \\
dataset  &              &              &              &              &              &               \\
\midrule
boston   &  -6.79±0.681 &  -4.24±0.304 &  -2.83±0.101 &   -7.61±2.31 &   -4.5±0.935 &  -13.09±1.207 \\
concrete &   -5.38±0.31 &  -6.19±1.566 &  -3.08±0.041 &  -3.62±0.143 &  -3.26±0.107 &    -3.6±0.217 \\
energy   &   -1.0±0.047 &  -1.79±0.281 &  -0.67±0.048 &  -1.31±0.128 &  -0.57±0.079 &   -0.87±0.209 \\
wine     &  -11.67±1.23 &  -2.35±0.193 &  -0.99±0.024 &   -1.2±0.041 &  -1.36±0.051 &   -1.45±0.066 \\
yacht    &  -0.59±0.034 &   -0.8±0.394 &  -1.65±0.124 &  -4.66±0.529 &  -0.97±0.264 &    -4.2±1.201 \\
\bottomrule
\end{tabular}
-->

## logs

![](Exp2Small_logs.png)

## Epistemic predictive variance distributions


### histogram
![](Exp2_small_EpistVarDist_Exp2-1.pdf)

scaled on variance range $[0,3]$.

![](Exp2_small_EpistVarDist_Exp2-scaled_1to3.pdf)



### empirical cumulative distribution
![](Exp2_small_EpistVarECDF_Exp2-1.pdf)


## ROC curves and AUC

![](Exp2_small_ROC_curves_VAR.pdf)

## Concrete detail

![](Exp2_small_EpistVarECDF_Exp2-concrete.pdf)![](Exp2_small_ROC_curves_concrete.pdf)


## Entropy of ensemble

For the ensemble method: 5 models, others 1K models. 
Parameter space entropy not available for MC dropoout.

### Parameter space

![](Exp2small_Predictor_Entropy.png)

<!--
\begin{tabular}{lllllll}
\toprule
{} & MC dropout &    Ensemble &        MFVI &      FuNN-MFVI &      NN-HyVI &     FuNN-HyVI \\
{} &  parameter &   parameter &   parameter &      parameter &    parameter &     parameter \\
\midrule
boston   &    nan±nan &  495.0±16.0 &  518.0±11.0 &  -1744.0±129.0 &  1259.0±25.0 &  -1317.0±62.0 \\
concrete &    nan±nan &  530.0±12.0 &  221.0±20.0 &  -1231.0±121.0 &   861.0±26.0 &   -975.0±39.0 \\
energy   &    nan±nan &  203.0±10.0 &  265.0±18.0 &  -1036.0±109.0 &   523.0±28.0 &  -1531.0±47.0 \\
wine     &    nan±nan &  933.0±19.0 &  445.0±11.0 &  -1232.0±106.0 &  1375.0±49.0 &   -994.0±69.0 \\
yacht    &    nan±nan &   169.0±9.0 &   236.0±8.0 &   -772.0±104.0 &   335.0±26.0 &   -936.0±40.0 \\
\bottomrule
\end{tabular}
-->

### Predictor space
![](Exp2small_Param_Entropy.png)

<!--
\begin{tabular}{lllllll}
\toprule
{} &   MC dropout &     Ensemble &          MFVI &    FuNN-MFVI &      NN-HyVI &    FuNN-HyVI \\
{} &    predictor &    predictor &     predictor &    predictor &    predictor &    predictor \\
\midrule
boston   &  -492.0±17.0 &  -103.0±18.0 &   -628.0±20.0 &  -412.0±31.0 &  -518.0±20.0 &  -241.0±10.0 \\
concrete &  -405.0±57.0 &  -154.0±10.0 &   -694.0±16.0 &  -395.0±30.0 &  -407.0±34.0 &   -259.0±4.0 \\
energy   &  -478.0±40.0 &  -298.0±33.0 &  -1146.0±12.0 &  -367.0±42.0 &  -382.0±50.0 &  -305.0±14.0 \\
wine     &  -234.0±19.0 &    50.0±13.0 &   -487.0±27.0 &  -332.0±14.0 &  -241.0±36.0 &   -250.0±6.0 \\
yacht    &  -812.0±12.0 &  -531.0±25.0 &  -1068.0±32.0 &  -346.0±38.0 &  -654.0±36.0 &   -281.0±4.0 \\
\bottomrule
\end{tabular}
-->

# EXP2 large

## RMSE

![](Large_RMSE.png)    
 
 <!---
\begin{tabular}{lllllll}
\toprule
{} &   MC dropout &     Ensemble &         MFVI &    FuNN-MFVI &      NN-HyVI &    FuNN-HyVI \\
dataset    &              &              &              &              &              &              \\
\midrule
navalC     &    0.001±0.0 &      0.0±0.0 &      0.0±0.0 &    0.001±0.0 &      0.0±0.0 &      0.0±0.0 \\
powerplant &  3.993±0.048 &  3.977±0.062 &  4.087±0.041 &  4.153±0.055 &  3.921±0.054 &  3.887±0.065 \\
protein    &  4.311±0.022 &  4.096±0.024 &  4.241±0.026 &  4.343±0.021 &   4.16±0.029 &  4.143±0.029 \\
\bottomrule
\end{tabular}
--> 
## LPP

![](Large_LPP.png)
<!--
\begin{tabular}{lllllll}
\toprule
{} &     MC dropout &       Ensemble &          MFVI &     FuNN-MFVI &       NN-HyVI &     FuNN-HyVI \\
dataset    &                &                &               &               &               &               \\
\midrule
navalC     &    5.919±0.126 &    6.121±0.121 &    6.27±0.093 &   6.129±0.044 &    7.033±0.05 &   7.055±0.054 \\
powerplant &   -15.865±0.55 &  -63.118±4.163 &  -2.827±0.011 &  -2.846±0.015 &  -2.786±0.014 &  -2.779±0.018 \\
protein    &  -20.019±0.587 &  -12.096±0.806 &  -2.864±0.006 &  -2.888±0.005 &  -2.845±0.007 &  -2.841±0.007 \\
\bottomrule
\end{tabular}
-->

## Logs

![](Lage_logs.png)

## Epistemic predictive variance distributions


### histogram
![](Large_EpistVarDist_Exp2.pdf)

### empirical cumulative distribution
![](Exp2_large_EpistVarECDF.pdf)

## ROC curves and AUC
![](Exp2_large_ROC_curves.pdf)

## Powerplant detail

![](Exp2_large_EpistVarECDF_Exp2-powerplant.pdf)
![](Exp2_large_ROC_curves_powerplant.pdf)


## Entropy of ensemble

For the ensemble method: 5 models, others 1K models. 
Parameter space entropy not available for MC dropoout.

### Parameter space
![](Param_Entropy_large.png)
<!--
\begin{tabular}{lllllll}
\toprule
{} & MC dropout &     Ensemble &          MFVI &      FuNN-MFVI &       NN-HyVI &        FuNN-HyVI \\
{} &  parameter &    parameter &     parameter &      parameter &     parameter &        parameter \\
\midrule
navalC     &    nan±nan &   644.0±15.0 &   -25.0±109.0 &  -5837.0±450.0 &   459.0±248.0 &  -12271.0±2147.0 \\
powerplant &    nan±nan &    272.0±5.0 &  -343.0±135.0 &  -1433.0±141.0 &    382.0±38.0 &    -1806.0±130.0 \\
protein    &    nan±nan &  1204.0±11.0 &  -350.0±146.0 &  -4038.0±336.0 &  1298.0±113.0 &     -3949.0±61.0 \\
\bottomrule
\end{tabular}
-->

### Predictor space
![](Predictor_Entropy_large.png)

<!--
\begin{tabular}{lllllll}
\toprule
{} &   MC dropout &     Ensemble &           MFVI &    FuNN-MFVI &      NN-HyVI &     FuNN-HyVI \\
{} &    predictor &    predictor &      predictor &    predictor &    predictor &     predictor \\
\midrule
navalC     &   -167.0±9.0 &    -71.0±7.0 &  -3698.0±225.0 &  -369.0±15.0 &  -779.0±34.0 &  -721.0±121.0 \\
powerplant &  -592.0±19.0 &  -608.0±14.0 &    -865.0±18.0 &  -431.0±29.0 &  -671.0±72.0 &   -337.0±22.0 \\
protein    &    -3.0±26.0 &    99.0±14.0 &    -472.0±22.0 &  -406.0±18.0 &  -213.0±29.0 &   -357.0±13.0 \\
\bottomrule
\end{tabular}
-->
