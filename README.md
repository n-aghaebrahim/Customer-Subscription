# Customer-Subscription
Predict Customer Subscription to Term Deposits using Bank Marketing Data Set.

# Bank Marketing Data Set for Customer Subscription

##Project Overview

This project is about predicting if a client will subscribe to a term deposit offered by a Portuguese banking institution. The data was obtained from direct marketing campaigns based on phone calls. The goal is to build a classification model that can predict whether or not a client will subscribe to the term deposit based on various attributes.

##Dataset Information

The dataset consists of four CSV files, with a total of 45,211 instances and 17 input features, as well as the output variable, which is whether the client subscribed to the term deposit or not. The input features include bank client data such as age, job type, marital status, education level, and more. It also includes data related to the last contact of the current campaign, such as the contact communication type, last contact month and day of the week, and last contact duration. Additionally, there are other social and economic context attributes such as the employment variation rate, consumer price index, consumer confidence index, euribor 3 month rate, and number of employees.

# Bank Marketing Dataset

This dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The goal is to predict if the client will subscribe a term deposit (variable y).

## Data Set Characteristics

- Multivariate
- Number of Instances: 45211
- Area: Business
- Attribute Characteristics: Real
- Number of Attributes: 17
- Date Donated: 2012-02-14
- Associated Tasks: Classification
- Missing Values? N/A
- Number of Web Hits: 2033891

### Source

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.

## Data Set Information

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

There are four datasets:

1. bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014].
2. bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3. bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4. bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).

The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

## Attribute Information

Input Variables:

- age (numeric)
- job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
- marital: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
- default: has credit in default? (categorical: 'no','yes','unknown')
- housing: has housing loan? (categorical: 'no','yes','unknown')
- loan: has personal loan? (categorical: 'no','yes','unknown')
- contact: contact communication type (categorical: 'cellular','telephone')
- month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
- day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
- duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.



#################################
##Model

The project is focused on building a classification model that can accurately predict whether a client will subscribe to a term deposit or not. Several machine learning algorithms can be applied to this problem, such as logistic regression, decision trees, random forests, and support vector machines. The performance of the models can be evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC curve.

##Usage

To use this project, you can download the dataset from the UCI Machine Learning Repository, and then preprocess and clean the data as necessary. You can then build and train your classification model using any suitable machine learning algorithm. Finally, you can evaluate the performance of your model and tune the hyperparameters as necessary to improve the accuracy and other metrics.

##License

This project is licensed under the [MIT License](LICENSE).



