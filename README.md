# League-Of-Legends-Result-Model-2023
This DSC80 project of UCSD involves tasks such as
data cleaning, feature engineering, correlation analysis, hypothesis testing, pipeline design, and building multiple regression and Random Forest models. Our goal is to distill insights from a complex dataset and present our findings on an interactive website.


Creators: Angelina Zhang and Ziyu Huang

## Introduction

We choose the dataset documenting professional League of Legends games throughout 2023; each game, uniquely identified by a distinct gameid shown in the first column, is represented with ten rows delineating individual player data for competing teams and an additional two rows offering comprehensive team statistics. 
The dataset boasts 129264 rows and 123 columns with detailed information about each game. 
Following the last project, this time, we focus on building a model that helps predict a game's result using not only the `side` column.


## Framing the Problem:
Prediction Problem and Type:
The core of our analysis revolves around a prediction problem classified as a binary classification task. Specifically, 
### we seek to predict whether a given game will result in a win or loss for a particular team with a random forest classifier. 
The response variable, or the variable we aim to predict, is the game's outcome `result`. In this context, we leverage data analytics to predict the outcome of a League of Legends (LoL) match, unraveling the intricate factors that contribute to success on the virtual battlefield. Our prediction model delves into features derived exclusively from in-game data. The key features under scrutiny encompass neutral resources (dragons, elders, heralds, barons) and team differentials (experience team differance at 10 and 15 minutes, damage per minute, gold difference at 10 and 15). And the information we used to provide valuable insight into the final result- victory or defeat are  

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
- `xpdiffat10`
- `xpdiffat15`
- `golddiff10`
- `golddiff15`
- `turretplates_diff`
- `natural_resource` (same calculation process as Side-Analysis-of-League-Of-Legends-2023 and we compute difference in natural resources to address the “time of prediction”  problem)

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
In the pursuit of accuracy, we opt for the Accuracy metric as the most suitable measure for our balanced dataset. With an equal distribution of wins and losses, and an equitable treatment of False Positives (FP) and False Negatives (FN), accuracy emerges as the preferred metric over F-1 score, precision, and recall.


## Baseline Model

### Exploratory Data Analysis (EDA)

In our exploratory data analysis, we noticed missing data correlated with the `league` variable. To address this, we utilized probabilistic imputation, leveraging data from other leagues to fill in missing values, ensuring a more comprehensive dataset.

To further understand the relationships between features, we conducted exploratory data analysis by drawing a scatter plot matrix. The color-coding is based on the game result ('Win' or 'Lose').

<iframe src="assets/scatterplt_correlation_columns.html" width=1000 height=800 frameBorder=0></iframe>

#### Observations:

1. Linear Correlations: There appears to be linear correlation between `golddiffat10`, `golddiffat15`, `xpdiffat10`, and `xpdiffat15`. To address potential multicollinearity, we plan to perform consider dropping certain columns in the next steps.
2. Distinguishable Cutoff: If a column (a) has a strong correlation (𝑟²=0.4) with all the other columns, column (a) might be represented as a linear combination of the rest of the columns. Therefore, we dropped such columns to reduce variance ($variance\propto d/n$, where d is number of columns and n is number of data points, or rows)
   
   This is the head of our result dataframe showing columns correlation with each other:

| target       |   r_squared | features                                     |     rmse |
|:-------------|------------:|:---------------------------------------------|---------:|
| golddiffat10 |    0.613964 | ['golddiffat15', 'xpdiffat10', 'xpdiffat15'] | 0.655504 |
| golddiffat15 |    0.54112  | ['xpdiffat10', 'xpdiffat15']                 | 0.726957 |
| xpdiffat10   |    0.328865 | ['xpdiffat15']                               | 0.934638 |
| xpdiffat15   |    0.328865 | ['xpdiffat10']                               | 0.925451 |
 
As depicted in the DataFrame above, our initial focus was on columns with an r-squared value greater than 0.4. Despite the promising nature of these features, the presence of a relatively high root mean squared error (rmse) prompted us to delve deeper into the analysis. To gain a clearer understanding of the predictive performance and potential issues, we proceeded to visualize the residuals, which allows us to scrutinize the disparities between the predicted values and the actual observations. This examination becomes particularly crucial when facing higher rmse values, as it helps identify patterns or trends that may not be evident through standard metrics alone.

<iframe src="assets/golddiffat10_residual.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/golddiffat15_residual.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/xpdiffat10_residual.html" width=800 height=600 frameBorder=0></iframe>



By plotting the residuals, we aimed to uncover any systematic deviations or patterns in our predictions. This step provides valuable insights into the limitations of our model and guides potential refinements to enhance its accuracy and reliability.



With the promising result of residual plot, we dropped `golddiffat10`, `golddiffat15`.



### Baseline Model Description
In our baseline model, we employed a logistic regression model using a preprocessor with One-Hot Encoding for the `side` variable. The features in the model include both quantitative and nominal variables:


Quantitative Features:
`dpm`
`xpdiffat10`
`xpdiffat15`


Nominal Features:
`side` (One-Hot Encoded)


In our baseline model, we selected specific features, including dpm, xpdiffat10, and xpdiffat15, for their potential influence on predicting the outcome of League of Legends games. These features were chosen based on their significance in reflecting key aspects of team performance during different phases of the game. dpm (damage per minute) provides insights into a team's offensive capabilities, while xpdiffat10 and xpdiffat15 represent the experience point differentials at 10 and 15 minutes, respectively, capturing the team's strategic advantage or disadvantage during crucial early and mid-game stages.

The rationale behind including these features lies in their ability to encapsulate critical moments and trends that significantly impact the overall game result. By focusing on the 10 and 15-minute time frames, we aim to capture the pivotal early and mid-game dynamics that often shape the trajectory of a match. These time points are strategically chosen to estimate the final result, allowing our model to account for key developments and performance differentials during critical phases of the game.

The decision to drop certain columns, such as golddiffat10 and golddiffat15, was guided by a comprehensive analysis of their linear correlations and root mean squared error (rmse) values. The scatter plot matrix and residual plots provided valuable insights into the predictive performance and potential issues, leading us to refine our model by retaining only the most informative features. This iterative process ensures that our logistic regression model is not only accurate but also effectively captures the nuances of League of Legends gameplay that contribute to the final outcome.



#### Feature Transformation and Hyperparameter Tuning

In this analysis, we employed a combination of feature transformation and hyperparameter tuning to enhance the performance of a logistic regression model. The feature transformation was executed using a preprocessor, specifically a ColumnTransformer, which applied a One-Hot Encoding transformation to the 'side' feature while preserving other features through the 'passthrough' option. This transformation is encapsulated within a Pipeline, along with the logistic regression model. To optimize the logistic regression model's performance, a grid search was conducted over the hyperparameter space, focusing on the max_iter parameter. The grid search, performed with cross-validation, identified the best-performing model with a max_iter value of 54. This parameter choice is supported by a graph depicting the model's performance across different max_iter values. The resulting tuned logistic regression model is expected to exhibit improved predictive capabilities, making it well-suited for the task at hand, which is shown below.

<iframe src="assets/bl_hyperpara_accuracy.html" width=800 height=600 frameBorder=0></iframe>

### Performance Evaluation
The model achieved a high accuracy of approximately 75.19%. The confusion matrix shows good performance in distinguishing between true positives (1623) and true negatives (1617), with fewer false positives (552) and false negatives (517).
Accuracy: 75.19%
Precision: 74.55%
Recall: 75.77%

Confusion matrix:
![image](https://github.com/Angelinaaaaaaaaaaaa/LOL-Result-Model-2023/assets/115201846/924ea8b0-6ff2-4e6d-8d37-42769bb0b030)



### Model Assessment
The baseline logistic regression model demonstrates strong predictive performance, as evidenced by its high accuracy and balanced precision and recall scores. The inclusion of key features, both quantitative and nominal, along with appropriate encodings, has contributed to the model's effectiveness.
The model's capability to correctly classify outcomes, as reflected in the confusion matrix, suggests its potential utility in predicting game results based on the specified features. While this baseline model performs well, further refinement and feature engineering may be explored in subsequent iterations to enhance its predictive capabilities.




## Final Model

### Feature Transformation

#### New Features

We enhanced our baseline model by incorporating additional features, including `doublekills`, `triplekills`, `quadrakills`, `pentakills`, `firstblood`, `firstdragon`, `elders`, `firstherald`, `firstbaron`, `firsttower`, `firstmidtower`, `firsttothreetowers`, `dpm`, `turretplates_diff`, and `natural_resource`. This augmentation aims to refine the final model.

The selection of features such as `turretplates_diff`, `firstblood`, `firstdragon`, and `firstherald` is based on their ability to reflect a team's early-game performance, contributing to the acquisition of initial advantages. Additionally, features related to `firstbaron`, `elder`, `doublekills`, `triplekills`, `quadrakills`, and `pentakills` were incorporated, as they consistently mark pivotal moments in the game, signifying significant shifts in momentum.

Furthermore, the inclusion of `firsttower`, `firstmidtower`, and `firsttothreetowers` serves a dual purpose by not only reflecting a team's mid-game performance but also shedding light on the vision disparities between the two teams. On another note, `dpm`, `turretplates_diff`, and `natural_resource` offer insights from a different perspective, providing indications of a team's consistency in performance throughout the game.


#### Hyperparameters:

To address our binary classification question(predict either win or lose), we opted to employ the RandomForest Classification (RFC) model on the dataset. Our primary focus during hyperparameter tuning revolves around three key parameters: the number of decision trees in RFC, the maximum depth of each decision tree, and the metric used for assessing disorder in each node — either entropy, which is measures of information gain, or Gini, which is the measure of impurity. Applying Cross validation for tuning the hyperparameters, we end up with 1250 decision trees, max_depth with 14, and the criterion is `gini`.

### Model Assessment
Here is a visualization of the confusion matrix after fitting the model:

 ![image](https://github.com/Angelinaaaaaaaaaaaa/LOL-Result-Model-2023/assets/115201846/33c0a9f8-bb60-49a5-95b9-5fbd1eec96f5)

Initially, upon examining the accuracy metrics, it's evident that the RandomForest Classification (RFC) model performs notably well, achieving an accuracy of approximately 86%. This represents a substantial improvement of 10% compared to the accuracy observed with the Logistic Regression model. Moreover, when evaluating precision and recall, the RFC model outperforms the baseline model. In summary, considering accuracy, precision, and recall collectively, it is apparent that RFC excels in effectively classifying the `result` variable.




## Fairness Analysis

### Group Selection:

For this fairness analysis, we selected Group X as the predictions made for games on the blue side and Group Y as the predictions made for games on the red side.

### Evaluation Metric:

Our evaluation metric is the accuracy of the classifier in predicting game outcomes for both Group X and Group Y.

### Hypotheses:

- **Null Hypothesis (H0):**
  The accuracy of the classifier is the same for both the blue side (Group X) and the red side (Group Y), and any observed differences are due to random chance.

- **Alternative Hypothesis (H1):**
  There is a significant difference in accuracy between the blue side (Group X) and the red side (Group Y).

### Test Statistic:

We utilized the absolute difference in accuracy between Group X and Group Y as our test statistic.

### Significance Level:

A significance level of 0.01 was chosen to determine the threshold for statistical significance.

### Permutation Test Results:

Upon conducting the permutation test, we obtained a p-value of 0.967. This p-value represents the probability of observing a difference in accuracy as extreme as the one observed, assuming the null hypothesis is true.
<iframe src="assets/Side_Accuracy_Difference.html" width=800 height=600 frameBorder=0></iframe>

### Conclusion:

Given our chosen significance level of 0.01, the obtained p-value of 0.967 exceeds the threshold. Therefore, we fail to reject the null hypothesis. This suggests that any observed variations in accuracy between the blue side and the red side may be attributed to random chance.

*Note: In the context of statistical tests, we do not make absolute conclusions. Our findings indicate that, under the conditions of our test, we do not have sufficient evidence to claim a significant difference in accuracy between predictions for games on the blue side and the red side.*

