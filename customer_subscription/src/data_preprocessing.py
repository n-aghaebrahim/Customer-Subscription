import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("bank-additional-full.csv", delimiter=";")

# Drop irrelevant columns
data = data.drop(["duration"], axis=1)

# Replace unknown values with NaN
data = data.replace("unknown", np.nan)

# Convert categorical variables to numerical
data["job"] = data["job"].astype("category").cat.codes
data["marital"] = data["marital"].astype("category").cat.codes
data["education"] = data["education"].astype("category").cat.codes
data["default"] = data["default"].astype("category").cat.codes
data["housing"] = data["housing"].astype("category").cat.codes
data["loan"] = data["loan"].astype("category").cat.codes
data["contact"] = data["contact"].astype("category").cat.codes
data["month"] = data["month"].astype("category").cat.codes
data["day_of_week"] = data["day_of_week"].astype("category").cat.codes
data["poutcome"] = data["poutcome"].astype("category").cat.codes
data["y"] = data["y"].astype("category").cat.codes

# Impute missing values with column means
data = data.fillna(data.mean())

# Split into features and target
X = data.drop(["y"], axis=1)
y = data["y"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#########################
########################
# Load the data
data = pd.read_csv('bank-additional-full.csv', sep=';')

# Remove the duration column
data = data.drop(['duration'], axis=1)

# Convert categorical variables to numerical using label encoding
label_encoder = LabelEncoder()
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Convert target variable to binary
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Split data into training and testing sets
x = data.drop(['y'], axis=1)
y = data['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



