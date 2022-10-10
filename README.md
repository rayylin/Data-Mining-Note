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
        Examining all possible models is infeasible, as there are 2p possible models.

        Forward selection: Begin with the null model - a model that contains an intercept but no predictors. Fit p simple linear regressions and add to the null model the variable that optimizes a selection criterion (e.g., lowest residual sum of squares (RSS)). In addition to it, add another variable that optimizes the
        selection criterion amongst all two-variable models. Continue until some stopping rule is satisfied, e.g., when all remaining variables have a p-value above some threshold. Drawback: each addition of a new variable may render one or more of the already included variables non-significant.

        Backward selection
        Start with all variables in the model. Remove the variable with the largest p-value - that is, the variable that is the least statistically significant.
        Continue until a stopping rule is reached. For instance, we may stop when all remaining variables have a significant p-value defined by some significance threshold. Drawback: sometimes variables are dropped that would be significant when added to the final reduced models.

        Shrinkage Methods: Lasso, Adaptive Lasso
        As an alternative, we can fit a model containing all p predictors using a technique that regularizes the coefficient estimates, or equivalently, that shrinks the coefficient estimates towards zero.

![image](https://user-images.githubusercontent.com/58899897/194777609-419ae755-af23-4723-a22d-9e9bf76377f5.png)

        As with least squares, lasso seeks coefficient estimates that fit the data well, by making the RSS small.
        However, the penalty term is small when (B1...Bp) are close to zero, and so it has the e↵ect of shrinking the estimates of Bj towards zero.

        Lambda is a tuning parameter and controls the balance of RSS and penalty. Lambda = 0 gives least squares estimator. if lambda -> infinite, thenB-Lasso -> 0
        Lasso forces some of the coefficient estimates to be exactly equal to zero when lambda is suffyciently large and hence lasso performs variable selection.

        Why can Lasso improve prediction accuracy?
        OLS estimator is unbiased, E(BˆOLS ) = B*(true parameter).
        B-lasso is a biased estimator, i.e., E(Bˆlasso) != B* if lambda != 0.
        But, Bˆlasso typically has a smaller prediction error than OLS.

        Bias-variance tradeo↵: When bias is low, variance will be high and vice-versa. A good model balances these two.
        The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations               between features and target outputs (underfitting).
        The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random           noise in the training data (overfitting).

![image](https://user-images.githubusercontent.com/58899897/194777984-1d1810c8-109c-4038-a181-82904ff1d4ea.png)

        Adaptive Lasso
        To reduce the bias in the lasso estimator, adaptive lasso imposes different penalties on the predictors

![image](https://user-images.githubusercontent.com/58899897/194778203-28ee8809-2284-4217-b0ef-abfc67cd8b2e.png)

        Larger BjOLS, less penalization, as larger Bj means more important
        Pros and Cons of Lasso, adaptive
        + : avoid overfitting of OLS: bias-variance tradeo↵s
        + : can handle p > n case
        + : has variable selection e↵ect
        -: no closed-form solution

CLogistic 7 Regression and Model Assessment in Classification:

        Logistic regression: one-dimension

![image](https://user-images.githubusercontent.com/58899897/194779858-38550fea-dbb5-40af-af5e-891875612360.png)

![image](https://user-images.githubusercontent.com/58899897/194779884-5c30603a-3752-45b8-9b1e-fabb3484be32.png)

        Input Coding: (class predictors)
        Deviation: constrains the parameters for all levels to sum to zero. Male = 1; Female = -1.
        GLM: use dummy variable. Male = 1; Female = 0.

![image](https://user-images.githubusercontent.com/58899897/194782046-4c5dbaca-0846-4b3a-8d02-793887e00fa3.png)

![image](https://user-images.githubusercontent.com/58899897/194782053-51f983af-ef7b-4453-8b6f-9609d84dac75.png)

![image](https://user-images.githubusercontent.com/58899897/194782097-29b97c27-6bf3-435b-a6d1-46e90bf1150d.png)

        Model assessment criterions for classification
        Misclassification rate, false negative, false positive
        Receiver operating characteristic (ROC) curve, AUC

        Misclassification rate

![image](https://user-images.githubusercontent.com/58899897/194782228-1c1f01a0-5bc0-4777-b286-89490866159a.png)

        Misclassification rate = 2.28 % +0.4 % = 2.68 %.
        False positive: The number of negative (No) examples that are classified as positive (Yes) = 40.
        False negative: The number of positive (Yes) examples that are classified as negative (No) = 228.
        False positive rate (FPR): The fraction of negative examples that are classified as positive = 40/9667 = 0.4138%
        False negative rate (FNR): The fraction of positive examples that are classified as negative = 228/333 = 68.4685%

        By default, logistic regression classifies a sample to class Yes if  Pb(default = Yes|balance,student, income) >= 0.5

![image](https://user-images.githubusercontent.com/58899897/194782379-9c63b630-151d-45a0-a354-5e4660ec51c2.png)

        When threshold = 0.2, FPR increases, FNR decreases, overall misclassification error increases, but the credit card company
        may consider this to be small price to pay for more accurate identification of customers who default!

        ROC & AUC

![image](https://user-images.githubusercontent.com/58899897/194782418-6e211b82-4261-4edb-b8b6-3066bf7d494d.png)
        The ROC plot displays both TPR (TPR = 1-FNR) and FPR simultaneously, summarized over all possible thresholds
        Sometimes we use the area under the curve (AUC) to summarize the overall performance. Higher AUC is better.


Chapter 8: Decision Tree

        Regression Tree

        Box Regions:
        Decision tree divides the predictor space, i.e., the set of possible values for X1,..., Xp, into J distinct and
        non-overlapping regions R1,..., RJ . It considers the shape of each Rj to be rectangles, or boxes.

        How to build a regression tree

![image](https://user-images.githubusercontent.com/58899897/194782706-a7d563c9-7d1d-40c3-a9e5-5740a4854c68.png)

        Recursive binary splitting: a top-down, greedy approach. The approach is top-down because it begins at the top of the
        tree and then successively splits the predictor space; each split is indicated via two new branches further down on the tree.
        It is greedy because at each step of the tree-building process, the best split is made at that particular step.

        We first select the predictor Xj and the cut point c such that splitting the predictor space into the regions {x |Xj < c} and
        {x |Xj c} leads to the greatest reduction in RSS (variance). Next, we repeat the process, looking for the best predictor
        and best cut point in order to split the data further so as to minimize the RSS within each of the child nodes,
        Again, we look to split one of these three regions further, so  as to minimize the RSS. The process continues until a stopping criterion is reached;
        Prediction: We predict the response for a given test  observation using the mean of the training observations in the region to which that test observation belongs.


        Classification Tree
        For a classification tree, we predict that each observation belongs to the most commonly occurring class of training observations in the region to which it belongs. RSS cannot be used as a criterion, as the variables are classes, rather than numbers.

        Gini index and entropy

![image](https://user-images.githubusercontent.com/58899897/194783307-6b0b7784-19e2-4e54-9585-0562ee4d45ee.png)

        The Gini index is small if all ˆpmk are close to zero or one.
        The Gini index is also referred to as a measure of node purity, that is, a small value indicates that a node contains mostly observations from a single class.

![image](https://user-images.githubusercontent.com/58899897/194783371-cbdb0053-7fd2-4db4-b932-d70049b68854.png)

![image](https://user-images.githubusercontent.com/58899897/194783387-2e60fabb-ada8-4221-b0c4-2bc5933ca8f1.png)

![image](https://user-images.githubusercontent.com/58899897/194783402-13fb3a71-e6e5-43e8-bd9d-91a9afc0f2b6.png)

        Pros and Cons of trees
        + Trees can be displayed graphically and are very easy to explain to non-technical people.
        + Trees can easily handle nominal predictors without the need to create dummy variables.
        + Trees can easily handle missing values.
        = However, trees generally do not have same level of predictive accuracy as some of other supervised learning methods.
        - Trees are very unstable: small variations in the data might result in a completely di↵erent tree being generated.
        
Chapter 9 Bagging, Random Forest, Boosting

        Bagging and Random Forest
        Bootstrap aggregation, or bagging, is a general-purpose procedure for reducing the variance of a data mining method.
![image](https://user-images.githubusercontent.com/58899897/194785245-639e818b-3e77-4c9c-b9e4-514d4522e88e.png)

        Bootstrap is to obtain distinct data sets by repeatedly sampling observations from the original data set with replacement.

        Bagging
![image](https://user-images.githubusercontent.com/58899897/194785334-5b4e62fd-a71b-4f11-a18b-5dcb3b822c1e.png)

        For classification trees: for each test observation x, we record the class predicted by each of the B trees, and take a majority vote: the overall prediction is the most commonly occurring class among the B predictions.

        In each bootstrapped set, observations from original training data that are not used for model fitting are out-of-bag samples. The out-of-bag error estimates the test error.

        Ramdom forests
        Random forests provide an improvement over bagged trees by  way of a small tweak that decorrelates the trees. This further reduces the variance when we average the trees. 
        As in bagging, we build a number of decision trees on bootstrapped training samples.
        Key: Decorelated. The price it has to pay is the increment of bias, which, fortunately, is usually small.

![image](https://user-images.githubusercontent.com/58899897/194785596-37c494b7-e57d-4382-95f8-bcdadcd05ff2.png)

        variable importance measure: OOB: Absolute Error or Valid: Absolute Error 
        The variables are important when their variable importance measure is positive.

        Boosting
        In Boosting, trees are grown sequentially: each tree is grown using information from previously grown trees.

        Unlike fitting a single large decision tree to the data, which amounts to fitting the data hard and potentially overfitting,
        the boosting approach instead learns slowly.
        Given the current model, we fit a decision tree to the residuals from the model. We then add this new decision tree into the
        fitted function in order to update the residuals. 
        Each of these trees can be rather small, with just a few terminal nodes, determined by tree depth in the algorithm.
        By fitting small trees to the residuals, we slowly improve prediction in areas where it does not perform well.

![image](https://user-images.githubusercontent.com/58899897/194785746-522a2d67-6058-419e-8b02-e3a62c6532aa.png)

        The shrinkage parameter slows the process down even further, allowing more and di↵erent shaped trees to attack the residuals.
        The smaller, the slower

![image](https://user-images.githubusercontent.com/58899897/194785826-2d439d5b-bde1-4d0a-9e90-6a4502df3bb3.png)

Chapter 10 Neural Networks:

![image](https://user-images.githubusercontent.com/58899897/194785925-988e2619-1134-4286-b4f4-2f68bbf358ad.png)

![image](https://user-images.githubusercontent.com/58899897/194785944-3f2cfc81-3a15-492c-8fb4-10128b3662ad.png)

Same layer, same activation and combination function.

![image](https://user-images.githubusercontent.com/58899897/194785994-a5db2f7b-335b-45da-8781-50f294069421.png)

![image](https://user-images.githubusercontent.com/58899897/194786023-91b80223-5e02-43ac-8cb8-faf30a081360.png)

![image](https://user-images.githubusercontent.com/58899897/194786039-7547bd6e-37f6-4739-8eb9-4b139dddf649.png)

![image](https://user-images.githubusercontent.com/58899897/194786114-a5a14780-6db1-41d7-b453-345256bd607d.png)

Chapter 11 K-means Clustering (Unsupervised learning)

![image](https://user-images.githubusercontent.com/58899897/194786503-13eeea83-f289-41ab-9749-b2ca8f104947.png)

For categorical variable: Use dummy

![image](https://user-images.githubusercontent.com/58899897/194786572-47af6147-b5f8-4b50-af24-50d7654ef12b.png)

Within Cluster Variation (WCV). We aim to minimize WCV

Steps of K-means
1. Initialization: randomly assign a number to each of the observations. These serve as initial cluster assignments.
2a For each of the K clusters, compute the cluster center ck as the sample mean of observations in the kth cluster.
2b Assign each observation to the cluster whose center is closest in Euclidean distance.
Iterate Step 2a&2b until cluster assignments stop changing.

Convergence: when the center is not moving or assignment does not change

![image](https://user-images.githubusercontent.com/58899897/194788198-9cb7f401-64a7-4783-af06-47a3a212ba49.png)

sensitive to initial points. in practice, we can assign different initial situations. choose the one with the smallest wcv

Kmeans uses multiple random initializations, and choose the one having smallest objective function value.
Any polynomial-time algorithm with global optimization

![image](https://user-images.githubusercontent.com/58899897/194788894-14b2f5d1-bd4a-408a-a0d8-85439fe7a2c3.png)

![image](https://user-images.githubusercontent.com/58899897/194788928-94ddb634-b559-4612-aa28-67f0fd42af43.png)

WCV -> 0 when increase K, as each point becomes a cluster.

Advantage and drawbacks of K-means Interview question!
+ Low memory usage
+ Fast algorithm
- Sensitive to random initialization
- Number of clusters is pre-defined
