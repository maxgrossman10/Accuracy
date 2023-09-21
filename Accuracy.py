# %% CREATE TWO NEAT CLUSTERS OF RANDOMLY GENERATED DATA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(0)

# Define the number of data points
num_data_points = 100

# Generate clustered data for 'No Arthritis' class
no_arthritis_center = [30, 2]  # Mean age and pain level for 'No Arthritis'
no_arthritis_std = [12, 2]  # Standard deviation for age and pain level
no_arthritis_data = np.random.normal(
    loc=no_arthritis_center, scale=no_arthritis_std, size=(num_data_points // 2, 2)
)
no_arthritis_labels = np.zeros(num_data_points // 2)

# Generate clustered data for 'Arthritis' class
arthritis_center = [60, 7]  # Mean age and pain level for 'Arthritis'
arthritis_std = [12, 2]  # Standard deviation for age and pain level
arthritis_data = np.random.normal(
    loc=arthritis_center, scale=arthritis_std, size=(num_data_points // 2, 2)
)
arthritis_labels = np.ones(num_data_points // 2)

# Combine data and labels
age = np.concatenate((no_arthritis_data[:, 0], arthritis_data[:, 0]))
pain_level = np.concatenate((no_arthritis_data[:, 1], arthritis_data[:, 1]))
arthritis = np.concatenate((no_arthritis_labels, arthritis_labels))

# Ensure age and pain level are non-negative
age = np.maximum(age, 0)
pain_level = np.maximum(pain_level, 0)

# Create a DataFrame to store the dataset
data = pd.DataFrame({"Age": age, "Pain_Level": pain_level, "Arthritis": arthritis})


###########################################################################################################
# VISUALIZE THE CLUSTERS BEFORE SPLITTING

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    data[data["Arthritis"] == 0]["Age"],
    data[data["Arthritis"] == 0]["Pain_Level"],
    label="No Arthritis",
    color="blue",
    alpha=0.6,
    s=80,
)
plt.scatter(
    data[data["Arthritis"] == 1]["Age"],
    data[data["Arthritis"] == 1]["Pain_Level"],
    label="Arthritis",
    color="orange",
    alpha=0.6,
    s=80,
)

# Set plot labels and title
plt.xlabel("Age")
plt.ylabel("Pain Level")
plt.title("Figure 2: Arthritis Classification Dataset with Clustered Data")

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


###########################################################################################################
# %% SPLIT THE DATA 80-20 TRAIN-TEST
from sklearn.model_selection import train_test_split

# Split the data into training (80%) and testing (20%) sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Display the sizes of the training and testing sets
print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))


###########################################################################################################
# %% VISUALIZE THE SPLIT DATA

import matplotlib.pyplot as plt

# Visualize both training and testing data points on a single plot
plt.figure(figsize=(10, 6))

# Training data (blue points)
plt.scatter(
    train_data[train_data["Arthritis"] == 0]["Age"],
    train_data[train_data["Arthritis"] == 0]["Pain_Level"],
    label="No Arthritis (Train)",
    color="blue",
    alpha=0.6,
    s=80,
)

plt.scatter(
    train_data[train_data["Arthritis"] == 1]["Age"],
    train_data[train_data["Arthritis"] == 1]["Pain_Level"],
    label="Arthritis (Train)",
    color="cyan",
    alpha=0.6,
    s=80,
)

# Testing data (orange points)
plt.scatter(
    test_data[test_data["Arthritis"] == 0]["Age"],
    test_data[test_data["Arthritis"] == 0]["Pain_Level"],
    label="No Arthritis (Test)",
    color="orange",
    alpha=0.6,
    s=80,
)

plt.scatter(
    test_data[test_data["Arthritis"] == 1]["Age"],
    test_data[test_data["Arthritis"] == 1]["Pain_Level"],
    label="Arthritis (Test)",
    color="red",
    alpha=0.6,
    s=80,
)

plt.xlabel("Age")
plt.ylabel("Pain Level")
plt.title("Figure 2:Training and Testing Data")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

###########################################################################################################
# %% CREATE A LOGISTIC REGRESSION CLASSIFICATION MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate the features (Age and Pain_Level) and the target variable (Arthritis) for training and testing sets
X_train = train_data[["Age", "Pain_Level"]]
y_train = train_data["Arthritis"]

X_test = test_data[["Age", "Pain_Level"]]
y_test = test_data["Arthritis"]

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)

# %%
# Assuming you have already trained your logistic regression model as 'model'
# Make predictions on the testing data
y_prob = model.predict_proba(X_test)

# y_prob will contain predicted probabilities for both classes (0 and 1)
# Extract the probabilities for class 1 (Arthritis)
probabilities_class_1 = y_prob[:, 1]

# Print the predicted probabilities for class 1
print("Predicted Probabilities for Class 1 (Arthritis):")
print(probabilities_class_1)
