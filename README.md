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
    A node for replacing possible outliers (Replacement node)
    A node for transforming skewed data (Transformation node)
    A node for performing regression or Decision Tree analysis  (Regression or Decision Tree node)
    A node for comparing the two models (Model Comparison node)



