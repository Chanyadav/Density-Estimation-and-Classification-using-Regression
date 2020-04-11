import scipy.io
import math
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal


# Calculate the mean for a particular image x
def mean(x):
    return sum(x) / float(len(x))

# Calculate the standard deviation of a given image a
def standard_deviation(a):
    average = mean(a)
    sum1 = 0
    for x in a:
        temp = x - average
        sum1 += temp * temp
        variance = sum1 / (len(a) - 1)
    return math.sqrt(variance)

# Extract the Feature 1 - (Mean) of an image x
def get_feature1_mean(x):
    tr_x = []
    for j in x:
        tr_x.append(mean(j))
    return tr_x
# Sigmoid function to constrain the output between 0 - 1
def sigmoid(x):
    """Sigmoide function"""
    return 1.0 / (1.0 + np.exp(-x))

# Extract the Feature 2 - (Standard Deviation) of an image x
def get_feature2_SD(x):
    tr_x = []
    for j in x:
        tr_x.append(standard_deviation(j))
    return tr_x

# Get the gaussian of a given sample, pair of (Mean and Standard Deviation)
def get_gaussians_distribution(x):
    gaussian = [mean(x), standard_deviation(x)]
    return gaussian

# Calculate the probability/likelihood of an input sample a part of a gaussian distribution
def calculate_probability(input_sample, average, sd, prior_probability):
    exponent = math.exp(-(math.pow(input_sample - average, 2) / (2 * math.pow(sd, 2))))
    return (1 / (math.sqrt(2 * math.pi) * sd)) * exponent * prior_probability

# Calculate the covariance matrix for a given digit
def print_covariance(classify_labels):
    temp = classify_labels
    t1 = temp[0]

    seven_cov = [[t1[0][1] * t1[0][1], 0], [0, t1[1][1] * t1[1][1]]]
    print("Covariance metrix for digit 7 ", seven_cov)
    t2 = temp[1]

    print()

    eight_cov = [[t2[0][1] * t2[0][1], 0], [0, t2[1][1] * t2[1][1]]]
    print("Covariance metrix for digit 8 ", eight_cov)

# Calculate the class probabilities for each of the labels (0 and 1)
def calculate_class_probability(gaussian_distribution, input_data, prior):
    class_probabilities = {}

    for x, classes in gaussian_distribution.items():
        class_probabilities[x] = 1
        for i in range(len(classes)):
            var = classes[i]
            # print(input_data[i])
            # print("Input of i")
            meanx = var[0]
            sdx = var[1]
            class_probabilities[x] *= calculate_probability(input_data[i], meanx, sdx, prior)
            # print("Probability of Class Value")
            # print(class_probabilities[x])
    return class_probabilities

# Classify which amongst the two classes have highest probability
def maximum_predict(classify_labels, input_test, prior):
    class_probabilities = calculate_class_probability(classify_labels, input_test, prior)

    out_label, max_prob = None, -1
    for class_value, probability in class_probabilities.items():
        if out_label is None or probability > max_prob:
            max_prob = probability
            out_label = class_value
    return out_label

# Calculate the accuracy of the entire dataset
def total_accuracy_entireDataset(predicts, test_label):
    correct = 0
    for x in range(len(test_label)):
        if test_label[x] == predicts[x]:
            correct += 1
    return (correct / float(len(test_label)))

# helper function to predict the probabilites of each label for a given input
def get_probabilitiy_estimate(classify_labels, test_data, prior):
    prediction_label = []
    for i in range(len(test_data)):
        result = maximum_predict(classify_labels, test_data[i], prior)
        prediction_label.append(result)
    return prediction_label

# Function to calculate the accuracy of each class
def get_accuracy(predicts, test_label):
    correct = 0
    for x in range(len(test_label)):
        if test_label[x] == predicts[x]:
            correct += 1
    return (correct / float(len(test_label)))


def naive_bayes():
    data_set = scipy.io.loadmat("C:/Users/Chandan Yadav/Downloads/mnist_data.mat")

    trainingData = data_set.get('trX')
    trainingDataLabel = data_set.get('trY')
    testingData = data_set.get('tsX')
    testingDataLabel = data_set.get('tsY')

    # get_feature1_mean calculates the mean for each sample in trainingData
    # trX1 is the mean for each image in the training dataset
    trX1 = get_feature1_mean(trainingData)

    # get_feature2_SD calculates the Standard Deviation for each sample in trainingData
    # trX2 is the Standard Deviation for each image in the training dataset
    trX2 = get_feature2_SD(trainingData)

    # tsX1 is the mean for each image in the testing dataset
    tsX1 = get_feature1_mean(testingData)

    # tsX2 is the standard deviation for each image in the testing dataset
    tsX2 = get_feature2_SD(testingData)

    # tsX is the list of all combinations of the features(Mean and SD)in the testing dataset
    tsX = list(zip(tsX1, tsX2))

    gaussian_distribution_7, gaussian_distribution_8 = [], []

    # get_gaussians_distribution calculates the mean and the SD for the given feature be it mean or standard_deviation for digit 7

    gaussian_distribution_7.append(get_gaussians_distribution(trX1[:6265]))
    gaussian_distribution_7.append(get_gaussians_distribution(trX2[:6265]))

    gaussian_distribution_8.append(get_gaussians_distribution(trX1[6265:]))
    gaussian_distribution_8.append(get_gaussians_distribution(trX2[6265:]))

# Classify the labels for the gaussian distribution into 0 and 1
    classify_labels = {0: gaussian_distribution_7, 1: gaussian_distribution_8}

    print_covariance(classify_labels)
# Calculate the prior probability of digit seven
    prior_seven = len(trX1[:6265]) / len(trX1)

    print("Prior probability of digit seven :", prior_seven)
    print()
# Calculate the prior probability of digit eight
    prior_eight = len(trX1[6265:]) / len(trX1)

    print("Prior probability of digit Eight :", prior_eight)
    print()

    predictions1 = get_probabilitiy_estimate(classify_labels, tsX[:1028], prior_seven)

    accuracy = get_accuracy(predictions1, testingDataLabel[0][:1028])
    print('Accuracy of classification using Naive Bayes Classifier for digit 7 is : ', accuracy * 100.0, ' %')

    predictions2 = get_probabilitiy_estimate(classify_labels, tsX[1028:], prior_eight)

    print()

    accuracy = get_accuracy(predictions2, testingDataLabel[0][1028:])
    print('Accuracy of classification using Naive Bayes Classifier for digit 8 is : ', accuracy * 100.0, ' %')

    print()

    finalAccuracy = total_accuracy_entireDataset(predictions1 + predictions2, testingDataLabel[0])
    print('Accuracy of the entire DataSet is', finalAccuracy * 100)


mat = scipy.io.loadmat("C:/Users/Chandan Yadav/Downloads/mnist_data.mat")
trainingData = mat.get('trX')
trainingDataLabel = mat.get('trY')
testingData = mat.get('tsX')
testingDataLabel = mat.get('tsY')
# Extract the feature 1 - Mean, of each image in the training dataset
feature1 = get_feature1_mean(trainingData)

# Extract the feature 2 - Standard Deviation, of each image in the training dataset
feature2 = get_feature2_SD(trainingData)

# Extract the feature 1 - Mean, of each image in the testing dataset
feature1_test = get_feature1_mean(testingData)

# Extract the feature 1 - Standard deviation, of each image in the testing dataset
feature2_test = get_feature2_SD(testingData)

# Initialize the parameters initially to zero
params_0 = np.zeros(3)

# Set the maximum number of iterations to 10000, check the accuracy for different values of max_iter.
max_iter = 50000

# New List consisting of all the features of training dataSet in a single list [1,x1,x2].
tr_new = []
for i in range(0, len(feature1)):
    tr_new.append([1, feature1[i], feature2[i]])

# New List consisting of all the features of testing dataSet in a single list [1,x1,x2].
ts_new = []
for i in range(0, len(feature1_test)):
    ts_new.append([1, feature1_test[i], feature2_test[i]])

ii = 0
# iterate through the maximum iterations to calculate the optimum number for the parameters
while ii < max_iter:

    multi_temp = []
    sigmoid_calc = []
    test_val = 0
# Multiply the parms_0 with the training dataset and append the result to a new list
    for i in range(len(tr_new)):
        total = 0
        for j in range(len(params_0)):
            total += params_0[j] * (tr_new[i][j])
        multi_temp.append(total)

# Use the sigmoid function to contain the result between 0 to 1
    for i in range(0, len(multi_temp)):
        test_val = sigmoid(multi_temp[i])
        sigmoid_calc.append(test_val)

    gard_temp = []
    temp_total = 0
# Calculate the Gradient by subtracting the sigmoid value with the trainingDataLabel
    for i in range(len(sigmoid_calc)):
        temp_total = trainingDataLabel[0][i] - sigmoid_calc[i]
        gard_temp.append(temp_total)

    featureX1 = []
    featureX2 = []
    featureX3 = []
# Extract the three features separately to calculate the parameters
    for i in range(len(tr_new)):
        for j in range(len(params_0)):
            if j == 0:
                featureX1.append(tr_new[i][j])
            if j == 1:
                featureX2.append(tr_new[i][j])
            if j == 2:
                featureX3.append(tr_new[i][j])
# Update the temporary values for the parameters each individually
    params_temp = []
    total = 0
    for i in range(len(featureX1)):
        total += featureX1[i] * gard_temp[i]

    params_temp.append(total)

    total = 0
    for i in range(len(featureX2)):
        total += featureX2[i] * gard_temp[i]
    params_temp.append(total)

    total = 0
    for i in range(len(featureX3)):
        total += featureX3[i] * gard_temp[i]
    params_temp.append(total)

# Set the value for alpha and multiply it with the gradient
    alpha = 0.0009

    for i in range(len(params_temp)):
        params_temp[i] = params_temp[i] * alpha

    for i in range(len(params_0)):
        params_0[i] = params_0[i] + params_temp[i]
    ii = ii + 1

prediction = []

# Calculating the predictions made by the model for each image in the testing set
for i in range(len(ts_new)):
    total = 0;
    for j in range(len(params_0)):
        total += (ts_new[i][j] * params_0[j])
    total_pred = sigmoid(total)
    if total_pred >= 0.5:
        prediction.append(1)
    if total_pred < 0.5:
        prediction.append(0)

# print(prediction)

# Calculate the accuracy of the model. If the model predicts more than 0.5 it's classified as 1 else less than 0.5 it's 0.
seven_cnt = 0
eight_cnt = 0

seven_total = 0
eight_total = 0

# Calculate the correct predictions of seven and eight respectively
for i in range(len(prediction)):
    if prediction[i] == testingDataLabel[0][i] and testingDataLabel[0][i] == 0:
        seven_cnt = seven_cnt + 1
    elif prediction[i] == testingDataLabel[0][i] and testingDataLabel[0][i] == 1:
        eight_cnt = eight_cnt + 1

    if testingDataLabel[0][i] == 0:
        seven_total = seven_total + 1
    else:
        eight_total = eight_total + 1
print("Accuracy for classifying digit seven-> ", (seven_cnt / seven_total) * 100,' %')
print("Accuracy for classifying digit eight-> ", (eight_cnt / eight_total) * 100,' %')

print("The accuracy of the Logistic Regression over the entire Data Set is : ", ( (seven_cnt+eight_cnt) / (seven_total+eight_total)) * 100,' %')
print()

naive_bayes()