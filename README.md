# Data-Mining-Note

Chapter 1 Introduction

    Supervised Learning 
    Two types of supervised learning:
    Regression: y is a Real number (stock, house price,traffic, Hurricane) 
    Linear model, Regression tree, ensembling(boosting, random forest), etc.

    Classification: y is a class {c1, c2,..., cK } (Spam detection, handwritten, microarray classification, cancer)
    Logistic Regression, Classification tree, ensembling(boosting, random forest), Neural Network

    Unsupervised Learning
    No outcome variable y, just a set of predictors x measured on a set of samples.

    Tasks:
        1.Clustering: find groups of samples that behave similarly, data points in one cluster are more “similar” to one another.
        2.Matrix factorization(因式分解): a factorization of a matrix into a product of smaller matrices.
    Ex: Market Segmentation: subdivide a market into distinct subsets of customers where any subset may conceivably be selected as a market 
    target to be reached with a distinct marketing mix.
    Ex: Video summarization:
    
Chapter 2: Intro to SAS EM    
    
    SAS Enterprise Miner Process for Data Mining: SEMMA
    1. Sample your data by extracting a portion of a large data set big enough to contain the significant information, yet small
    enough to manipulate quickly.
    2. Explore the data by searching for anticipated relationships, unanticipated trends, and anomalies in order to gain understanding and ideas.
    3. Modify the data by creating, selecting, and transforming the variables to focus the model selection process.
    (Data Inconsistency: IN, Indiana Transform. Scaling: scale Data that are not comparable)
    4. Model the data by using the analytical tools to search for a combination of the data that reliably predicts a desired outcome.
    5. Assess the data by evaluating the usefulness and reliability of the findings from the data mining process.

    In order to access the SAS data files using SAS EM, we mustcreate a SAS library. When you create a library, you give SAS a shortcut name and
    pointer to a storage location in your operating environment where you store SAS files(sas7bdat).

    Model Role:
        Input: independent variable, predictor
        Target: dependent variable, response
        ID: observation ID.
        Rejected: variables not used in the model
    Measurement Level:
        Binary: binary variable
        Nominal: categorical variable with no ordering
        Ordinal: categorical variable with ordering
        Interval: continuous variable

    A node for exploratory data analysis (StatExplore node)
    A node for filling missing values (Impute node)
    A node for replacing possible outliers (Replacement node) *Any observations that are more than 1.5 IQR below Q1 or more than 1.5 IQR above Q3 are considered outliers. 
    A node for transforming skewed data (Transformation node)
    A node for performing regression or Decision Tree analysis  (Regression or Decision Tree node)
    A node for comparing the two models (Model Comparison node)
    
Chapter 3 Exploratory Data Analysis (EDA)

        Use simple arithmetic and easy-to-draw graphs
        to gain insight into a data set:
                Special data pattern
                Center and variation of data
                Association/correlations between variables
                Important variables for later formal analysis
        to check quality of the data:
                Outliers
                Missing values
                Distribution of data
![image](https://user-images.githubusercontent.com/58899897/194739651-1429886d-5db2-4f74-a68d-9e8cb3f22338.png)

![image](https://user-images.githubusercontent.com/58899897/194743528-277f8f86-7792-4f87-aaf2-a6265cce6e68.png)

![image](https://user-images.githubusercontent.com/58899897/194743531-002ff66b-645a-4100-884f-a00de432f7d4.png)

How to do with continous number? Binning e.g.: 5bins

        Chi-Square: between two categorical (class) variables
        Chi-square is a measure of association of an input variable with the class target variable.
        The larger the Chi-square, the higher degree of association with class target.
        When target is an interval variable, correlations with interval input variables are also useful.
        Oij = observed freq for cell
        Eij = Expected freq for cell, if independent
                r        c
        X^2 = Sigma   (Sigma  ((Oij-Eij)^2/Eij))  with df = (r-1)(c-1)
               i = 1   j = 1

        X^2 statistic measures deviations from independence.
        A larger X^2 indicates a stronger evidence against H0, hence we have a higher degree of association.
        If chi square value = 0 -> p value = 1 -> cannot reject
        

![image](https://user-images.githubusercontent.com/58899897/194743636-634436fa-488d-4ec6-8914-05584e4efb80.png)

        p-value is a statistical measurement used to validate a hypothesis against observed data. A p-value measures the probability of obtaining the observed results, assuming that the null hypothesis is true. The lower the p-value, the greater the statistical significance of the observed difference.
        A p-value measures the probability of obtaining the observed results, assuming that the null hypothesis is true.
        
        p值是基於數據的檢定統計量算出來的機率值。如果p值是5%，也就是說，如果以此為界拒絕虛無假說的話，那麼只有5%的可能性犯錯。虛無假說是對的，但卻拒絕了，這是錯誤的。所以說p值越大，拒絕虛無假說的理由越不充分。如果p值接近於0，拒絕虛無假說，那麼幾乎不可能犯錯，於是說明數據是極其不符合虛無假說。

# Chapter 4 Data Preprocessing

        Detect and Handle Errors and Outliers
        Z-score and Box plot
        We can use Filter or Replace Node to handle outliers in SAS EM

        Filter: ignores Target and rejected Inputs.
            Rare Values (Count): Drop rare levels that have a count/frequency less than Minimum Frequency Cuto↵ (to be specified by user).
            So does Rare Values (percentage)
        Mehotd of Filter:
            Median Absolute Deviation (MAD): Eliminate values outside Median ± n * MAD. n = 9 by default
            MAD(D) = median(|di - median(D)|) 
            For example:   1,1,2,2,4,6,12
            Difference(di) 1,1,0,0,2,4,10 -> sorted 0,0,1,1,2,4,10 -> 1
            2 +- 1 * 9 -> (-7,11)
            Standard Deviations from the Mean: Eliminate values outside Mean±n⇥SD. EM default n = 3.

        Replacement: Replace specific non-missing values and keep replaced cases.
        Replacement node to generate score code to process unknown levels when scoring and also to interactively specify replacement values for class and interval levels. In some cases you may want to reassign specified nonmissing values (trim your variable's distribution) before performing imputation calculations for the missing values. To create a tighter and more centralized distribution. Replace before imputing missing values.

        Replace can be used to handle data inconsistency (IN, Indiana)

        How to Handle Missing Data?
        Ignore entire row/case may have high variance in missing values per attribute
        Use global constant to fill in missing value
        Use attribute mean to fill in missing value
        Use attribute mean for all samples belonging to same class to fill in missing value: smarter
        Use most probable value (mode) to fill in missing value
        Use a distribution-based method

        class variable: count, default constant, distribution, tree, tree surrogate
        Interval variable: mean, median, midrange, distribution, tree, tree surrogate, mid-minimum spacing, huber, andrew's wave, default constant
        Tree method: node, a variable with missing values will be treated as the target and all other input variables as the predictors.
        The predicted level/value will be used to fill the missing value.

        Data Transformation (Impute missing values before you transform variables.)
                To improve the fit of a model to the data.
                To stabilize variances (by transforming the target).
                To handle nonlinearity.
                To correct non-normality or skewness in variables.
                To transform class variables: dummy variables
                To create interaction variables.
                To prevent a variable (with a large variance) from dominating
                the others; i.e., to standardize variables.

        Transformation methods:
        Simple: Log, Square root, exp, standardize
        BinningL Bucket, Optimal binning
        Best Power Transformations(Try all and select best R^2):Maximize normality, Maximize correlation with target

Chapter 5: Linear Regression and Model Assessment

        Multiple linear regression
        X is predictor, and Y is response.

                     p
        Yi = B0 + (sigma Bj Xij)+ e  
                    j=1
        e.g.: sales = B0 +B1*TV + B2* radio + B3*newspaper + e
        The matrix form of multiple linear regression y = X*B +e

        Ordinary Least Squares

        ![image](https://user-images.githubusercontent.com/58899897/194751405-043a3e1c-2528-4b42-ab8f-84dbf0dc08d4.png)

        ![image](https://user-images.githubusercontent.com/58899897/194773912-8c4be219-8c8a-45e9-a60f-f82f12f79165.png)

        Why is it contradict with simple linear regression (Fit “sales” using only “newspaper” as predictor)?
        Correlation between radio and newspaper
        Confounding effect: the effects of the exposure under study on a given outcome are mixed in with the effects of an additional factor (or set of factors) resulting in a distortion of the true relationship.
        Confounder: a variable influences both dependent variables.

        ![image](https://user-images.githubusercontent.com/58899897/194774175-79988d2d-7fb8-436d-abf9-62e2db1c37e8.png)

        The coefficient 6 of TV*radio indicates the existence of synergy e↵ect: spending money on radio advertising actually increases the e↵ectiveness of TV advertising.

        Model assessment
        R Square: (TSS-RSS)/TSS = 1 - RSS/TSS
        RSS (Residual Sum Square): Sigma (Yi - Yi-hat)^2
        TSS (Total Sum Square): Sigma (Yi - Y-bar)^2
        R^2 measures the proportion of variability in y that can be explained using x.
        R^2 is always non-decreasing (RSS is non-increasing) as the number of variables in the model increases.
        The adjusted R2
        Adjusted R^2 = 1 - (RSS/(n-d-1)) / (TSS/(n-1))
        Nn is the number of observations in your data sample.
        d is the number of independent regressors, i.e. the number of variables in your model, excluding the constant.

        The AIC (Akaike information criterion) is defined for models fit by maximum likelihood
        AIC = 2 ln(L) + 2*d
        AIC = n*ln(RSS/n) + 2*d (in Normal case)
        where L is the maximized likelihood function for the estimated model, d is the total number of parameters used.

        The Schwarz’s Bayesian Criterion SBC, aka, BIC
        SBC = 2 ln(L) + ln(n)*d
        SBC = n ln(RSS/n) + ln(n)*d (in Normal case)

        SBC places a heavier penalty on models with many variables, and hence results in the selection of smaller models than AIC
        Selection rule: select the model with smallest AIC, SBC (BIC), or largest adjusted R2.

        Test Error
        Ideally, the optimal model should have the smallest test error. Test error is not available in practice! Need to estimate it!
        The RSS is a training error, computed on the training data. A model with zero training error typically is overfitting.

        We can estimate the test error, using either a validation set approach or a cross-validation approach.
        This procedure has advantages relative to AIC, BIC, and adjusted R2:
                provide a direct estimate of the test error
                doesn’t require the estimation to be likelihood-based or error         to be Gaussian
                doesn’t need to compute degrees of freedom

        ![image](https://user-images.githubusercontent.com/58899897/194776456-2eee8000-9fb3-4ad2-8e2e-dc3f10597415.png)

        Cross-validation
        In data insu

        cient situation, we can use cross-validation for both model selection and test error estimation
        1.Randomly split the training data into K roughly equal parts
        2.For each k = 1, ...,K
                a. fit the model using the other K-1 parts of the data.
                b. calculate prediction error of the fitted model in 2a on the kth part of the data.
        3.The averaged error over K prediction errors is the cross-validation error.
        In regression, the prediction error is the mean squared error (MSE). (1/n) * (sigma (yi - y-hat)^2)

Chapter 6 Subset Selection, Lasso, Adaptive Lasso

Why consider alternatives to least squares?
Prediction Accuracy: control the variance to reduce the mean square error (MSE). The least squares will fail when p > n.
Model Interpretability: By removing irrelevant features, i.e., by setting the corresponding coefficient estimates to zero, we can obtain a more interpretable model. We will present some approaches for automatically performing variable selection.

Subst selection

