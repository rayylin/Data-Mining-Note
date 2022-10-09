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




