# League-Of-Legends-Result-Model-2023
This DSC80 project of UCSD involves tasks such as
data cleaning, feature engineering, correlation analysis, hypothesis testing, pipeline design, and building multiple regression and Random Forest models. Our goal is to distill insights from a complex dataset and present our findings on an interactive website.


Creators: Angelina Zhang and Ziyu Huang

## Introduction

We choose the dataset documenting professional League of Legends games throughout 2023; each game, uniquely identified by a distinct gameid shown in the first column, is represented with ten rows delineating individual player data for competing teams and an additional two rows offering comprehensive team statistics. 
The dataset boasts 129264 rows and 123 columns with detailed information about each game. 
Following the last project, this time, we focus on building a model that helps predict a game's result using not only the `side` column.


Prediction Problem and Type:
The core of our analysis revolves around a prediction problem classified as a binary classification task. Specifically, we seek to predict whether a given game will result in a win or loss for a particular team with a random forest classifier. The response variable, or the variable we aim to predict, is the game's outcome `result`. And the information we used to provide valuable insight into the final result- victory or defeat are  

- `doublekills`
- `triplekills`
- `quadrakills`
- `pentakills`
- `firstblood`
- `firstdragon`
- `elders`
- `firstherald`
- `firstbaron`
- `firsttower`
- `firstmidtower`
- `firsttothreetowers`
- `dpm`
- `xpdiffat15`
- `turretplates_diff`
- `natural_resource` (same calculation process as last project and we compute difference in natural resources to address the “time of prediction”  problem)

,which can all be obtained in the process of a game.

This is the first five lines of our initial dataframe:

| side   |   doublekills |   triplekills |   quadrakills |   pentakills |   firstblood |   firstdragon |   elders |   firstherald |   firstbaron |   firsttower |   firstmidtower |   firsttothreetowers |     dpm |   golddiffat10 |   golddiffat15 |   xpdiffat10 |   xpdiffat15 |   result |   turretplates_diff |   natural_resource |
|:-------|--------------:|--------------:|--------------:|-------------:|-------------:|--------------:|---------:|--------------:|-------------:|-------------:|----------------:|---------------------:|--------:|---------------:|---------------:|-------------:|-------------:|---------:|--------------------:|-------------------:|
| Blue   |             1 |             1 |             0 |            0 |            0 |             0 |        0 |             1 |            1 |            1 |               1 |                    1 | 2186.9  |             75 |           -530 |         -156 |        -1671 |        1 |                   2 |                  3 |
| Red    |             1 |             0 |             0 |            0 |            1 |             1 |        0 |             0 |            0 |            0 |               0 |                    0 | 1960.18 |            -75 |            530 |          156 |         1671 |        0 |                  -2 |                 -3 |
| Blue   |             2 |             0 |             0 |            0 |            0 |             0 |        1 |             1 |            1 |            0 |               1 |                    1 | 2623.79 |           -361 |            673 |          282 |          530 |        0 |                   4 |                 -0 |
| Red    |             1 |             0 |             0 |            0 |            1 |             1 |        0 |             0 |            0 |            1 |               0 |                    0 | 1979.51 |            361 |           -673 |         -282 |         -530 |        1 |                  -4 |                  0 |
| Blue   |             2 |             2 |             0 |            0 |            0 |             0 |        0 |             0 |            0 |            0 |               1 |                    0 | 1968.55 |          -1001 |          -1901 |        -1748 |         -763 |        1 |                  -3 |                  1 |

By employing these evaluation metrics—accuracy, precision, and recall—using confustion matrix we aim to comprehensively assess the effectiveness of our Random Forest model in predicting League of Legends game outcomes.


## Baseline Model

### Exploratory Data Analysis (EDA)

In our exploratory data analysis, we noticed missing data correlated with the 'league' variable. To address this, we utilized probabilistic imputation, leveraging data from other leagues to fill in missing values, ensuring a more comprehensive dataset.

To further understand the relationships between features, we conducted exploratory data analysis by drawing a scatter plot matrix. The color-coding is based on the game result ('Win' or 'Lose').

<iframe src="assets/scatterplt_correlation_columns.html" width=800 height=600 frameBorder=0></iframe>

#### Observations:
1. Linear Correlations: There appears to be linear correlation between 'golddiffat10', 'golddiffat15', 'xpdiffat10', 'xpdiffat15', and 'turretplates_diff'. To address potential multicollinearity, we plan to perform log transformation and consider dropping certain columns in the next steps.
2. Distinguishable Cutoff: All variables seem to have a distinguishable cutoff. If a column (a) has a strong correlation (𝑟²=0.4) with all the other columns, column (a) might be represented as a linear combination of the rest of the columns. Therefore, we dropped such columns to reduce variance.
   
   This is the head of our result dataframe showing columns correlation with each other:


| target            |   r_squared | features                                                                                     |     rmse |
|:------------------|------------:|:---------------------------------------------------------------------------------------------|---------:|
| golddiffat10      |    0.615094 | ['golddiffat15', 'xpdiffat10', 'xpdiffat15', 'turretplates_diff', 'dpm', 'natural_resource'] | 0.645081 |
| golddiffat15      |    0.679717 | ['xpdiffat10', 'xpdiffat15', 'turretplates_diff', 'dpm', 'natural_resource']                 | 0.598195 |
| xpdiffat10        |    0.411069 | ['xpdiffat15', 'turretplates_diff', 'dpm', 'natural_resource']                               | 0.829502 |
| xpdiffat15        |    0.309912 | ['turretplates_diff', 'dpm', 'natural_resource']                                             | 0.93868  |
| turretplates_diff |    0.33633  | ['xpdiffat15', 'dpm', 'natural_resource']                                                    | 0.92878  |

Using the 0.4 cutoff of r_squared, we dropped `golddiffat10`, `golddiffat15`, `xpdiffat10`.


### Baseline Model Description
In our baseline model, we employed a logistic regression model using a preprocessor with One-Hot Encoding for the 'side' variable. The features in the model include both quantitative and nominal variables:
Quantitative Features:

`dpm`
`xpdiffat15`
`turretplates_diff`
`natural_resource`

Nominal Features:
`side` (One-Hot Encoded)
`doublekills`
`triplekills`
`quadrakills`
`pentakills`
`firstblood`
`firstdragon`
`elders`
`firstherald`
`firstbaron`
`firsttower`
`firstmidtower`
`firsttothreetowers`

The logistic regression model was tuned using a grid search with the parameter max_iter, and the best-performing model was obtained with max_iter=400.

### Performance Evaluation
The model achieved a high accuracy of approximately 85.97%. The confusion matrix shows good performance in distinguishing between true positives (1819) and true negatives (1840), with fewer false positives (357) and false negatives (293).
Accuracy: 85.92%
Precision: 83.75%
Recall: 86.26%

Confusion matrix:

![image](https://github.com/Angelinaaaaaaaaaaaa/LOL-Result-Model-2023/assets/115201846/19bf1ae5-3bcf-4b84-9b0c-8d580e55d92a)

### Model Assessment
The baseline logistic regression model demonstrates strong predictive performance, as evidenced by its high accuracy and balanced precision and recall scores. The inclusion of key features, both quantitative and nominal, along with appropriate encodings, has contributed to the model's effectiveness.
The model's capability to correctly classify outcomes, as reflected in the confusion matrix, suggests its potential utility in predicting game results based on the specified features. While this baseline model performs well, further refinement and feature engineering may be explored in subsequent iterations to enhance its predictive capabilities.




## Final Model

## Fairness Analysis


## Final Model

## Fairness Analysis
