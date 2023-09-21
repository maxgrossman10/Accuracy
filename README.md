# AccuracyAccuracy & Classification 
Accuracy = (# Correct Predictions)/( # Total Predictions) and is the most common metric used to evaluate the performance of classification models, which is predicting the correct label for input data. 
To use accuracy as a metric for a classification model, the dataset should be balanced, meaning there’s roughly an equal number of data points for each class.

For example, does the new data point belong to Class A or Class B in the figure? After we assign the new data point to either A or B, we would gauge the performance using accuracy in this fashion: 
in a supervised model, the test set would have the answer (ground truth), we’ll say A; if our model chose B, then our accuracy would be 0, and we would have a false negative. 
Let’s discuss the prediction errors false positives and negatives because in classification problems, it can be more important to minimize one over the other. 


Let’s first look at false positives. Assume that the red star is a new patient, and our model diagnoses them as having arthritis when, in reality, they don’t (Figure 1). 
This is a false positive, a Type 1 error. If instead our patient did have arthritis like the green star, and they were classified as not having arthritis, this would be a false negative, or Type II error. 

Returning to accuracy, let’s create a classification model and measure its performance on a variation of the same dataset. 
 

First, we’ll conduct an 80-20 train-test split with our data. 

 

When we create our classification model, we’re trying to predict where our testing data goes, based on age and pain level.
When we predict something, we’re using probability, and therefore want a logistic regression model. 
Logistic regression models the probability that an input belongs to a particular class using a logistic (sigmoid) function. 
Logistic regression produces an output in the range [0, 1], which can be interpreted as a probability. 
This is well-suited for binary classification, where the predicted probability can be compared to a threshold (e.g., 0.5) to make a class assignment. 
Our model predicted the class of 20 test observations and produced the following results. 

 
 

Each value in the list corresponds to the predicted probability that a test data point belongs to Class 1 (Arthritis), and an accuracy of 95% tells us that only 1 out of 20 predictions was wrong. 
