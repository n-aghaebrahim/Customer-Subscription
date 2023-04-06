import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('bank.csv')

# Create a new feature: age group
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 50, 60, np.inf], 
                           labels=['20s', '30s', '40s', '50s', '60+'])

# Create a new feature: education level
data['edu_level'] = data['education'].replace({'basic.4y': 'basic', 'basic.6y': 'basic', 'basic.9y': 'basic',
                                               'high.school': 'high_school', 'professional.course': 'higher',
                                               'university.degree': 'higher', 'illiterate': 'unknown'})

# Create a new feature: contact frequency
data['contact_freq'] = data.groupby(['campaign', 'poutcome'])['pdays'].transform('count')

# Drop irrelevant columns
data = data.drop(['age', 'education'], axis=1)

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month',
                                     'day_of_week', 'poutcome', 'age_group', 'edu_level'], drop_first=True)

# Split data into training and testing sets
X = data.drop('y', axis=1)
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

