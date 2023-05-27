# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Guilherme Lawnrence Rebouças Oliveira

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Beacause of the regression characteristics of the model, some predictions were slightly negatives.

### What was the top ranked model that performed?
LightGBM_BAG_L1/T1 with the selected non-defaul hyperparameters.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Analysing the data is possible to identify some behaviour that is specific to the day of the week, so i tried adding this information as a new feature to the model.

### How much better did your model preform after adding additional features and why do you think that is?
It performed 46% better, decreassing by almost half the score on kaggle. It happened beacause people usualy demand bikes on the same weekdays.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
Increased aditionaly 18% better beeing 55% better then the first one.

### If you were given more time with this dataset, where do you think you would spend more time?
I wold spend more time analysing the data in order to add new relevant parameters.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|presets|time_limit| NN_TORCH |score|
|--|--|--|--|--|
|initial|best_quality|60|defalts|1.30209|
|add_features|best_quality|60|defalts|0.72863|
|hpo|best_quality|60|num_epochs = 10| 0.59377|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](model_test_score.png)

## Summary

In this project, I used AutoGluon to predict bike sharing demand. I added the day of the week as a new feature, resulting in a 46% improvement in model performance. I also tried different hyperparameters and achieved an additional 18% improvement. If given more time, I would analyze the data to identify and add new relevant parameters. Overall, I developed a successful solution with significant improvements in model performance.