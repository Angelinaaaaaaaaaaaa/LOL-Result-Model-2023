# League-Of-Legends-Result-Model-2023
This DSC80 project of UCSD involves tasks such as
data cleaning, feature engineering, correlation analysis, hypothesis testing, pipeline design, and building multiple regression and Random Forest models. Our goal is to distill insights from a complex dataset and present our findings on an interactive website.


Creators: Angelina Zhang and Ziyu Huang

## Introduction

We choose the dataset documenting professional League of Legends games throughout 2023; each game, uniquely identified by a distinct gameid shown in the first column, is represented with ten rows delineating individual player data for competing teams and an additional two rows offering comprehensive team statistics. 
The dataset boasts 129264 rows and 123 columns with detailed information about each game. 
Following last project, this time, we focus on building a model that helps predicting the result of a game using not only `side` column.

We choose the dataset documenting professional League of Legends games throughout 2023; each game, uniquely identified by a distinct gameid shown in the first column, is represented with ten rows delineating individual player data for competing teams and an additional two rows offering comprehensive team statistics. 
The dataset boasts 129264 rows and 123 columns with detailed information about each game. 
Following the last project, this time, we focus on building a model that helps predict a game's result using not only the `side` column.


Prediction Problem and Type:
The core of our analysis revolves around a prediction problem classified as a binary classification task. Specifically, we seek to predict whether a given game will result in a win or loss for a particular team with a random forest classifier. The response variable, or the variable we aim to predict, is the game's outcome. And the information we used to provide valuable insight into the final result- victory or defeat are  

'xpdiffat15',
 'turretplates_diff', 
'dpm', 
and 'natural_resource,'(difference in naturals) 

which can be obtained in the process of a game.

By employing these evaluation metrics—accuracy, precision, and recall—we aim to comprehensively assess the effectiveness of our Random Forest model in predicting League of Legends game outcomes.
