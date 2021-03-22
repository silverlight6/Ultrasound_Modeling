import pandas as pd
import numpy as np
import random

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


# import data
dataPath = '/home/silver/Documents/TBI_NNs/Datasets/CSVFiles/patient12.csv'
axesDataPath = '/home/silver/Documents/TBI_NNs/Datasets/CSVFiles/xyAxis.csv'
data = pd.read_csv(dataPath)
axes = pd.read_csv(axesDataPath)

# start for loop here.
# while R.remaining:
# split into train/test sets
# trainX, testX, trainY, testY = train_test_split(X, y, test_size=(2/30), random_state=2*i)
# j = random.randint(1, 30)
j = 20
k = 24
# k = random.randint(1, 30)
# Check for the same value
while j == k:
    k = random.randint(1, 30)

# Separate the train set
trainX = data.where(~((data['fileID'] == j) | (data['fileID'] == k)))
# Separate the test sets individually because they are of different lengths
testXA = data.where(data['fileID'] == j)
testXB = data.where(data['fileID'] == k)
trainX = trainX.dropna(axis=0, how='all')
testXA = testXA.dropna(axis=0, how='all')
testXB = testXB.dropna(axis=0, how='all')
trainY = trainX['class'].values.reshape(-1, 1)
# Separate the test values individually
testYA = testXA['class'].values.reshape(-1, 1)
testYB = testXB['class'].values.reshape(-1, 1)
trainX = trainX.drop(['class', 'fileID'], axis=1)
testXA = testXA.drop(['class', 'fileID'], axis=1)
testXB = testXB.drop(['class', 'fileID'], axis=1)
trainY = trainY.ravel()
# I wish I understood these 2 lines but I think they are needed for roc curves
ns_probA = [0 for _ in range(len(testYA))]
ns_probB = [0 for _ in range(len(testYB))]

# fit a model
model = QuadraticDiscriminantAnalysis()
model.fit(trainX, trainY)

# used in the true images
xAxis = axes['row']
yAxis = axes['col']
zeroM = axes['zerosMatrix']
zeroM = zeroM + 1
# used in the first prediction
xAxisA = testXA['row']
yAxisA = testXA['col']
# used in the second prediction
xAxisB = testXB['row']
yAxisB = testXB['col']

i = .5
r = 0
c = 0
plt.style.use('seaborn-dark-palette')
fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
# this iterates 8 times.
while r < 4:
    # Below this point is graphing code
    probA = model.predict(testXA)
    probB = model.predict(testXB)

    # Threshold changes
    probA = np.where(probA > i, 1, 0)
    probB = np.where(probB > i, 1, 0)

    axs[r, c].scatter(yAxis, xAxis, zeroM, color='#000000')
    axs[r, c].scatter(yAxisA, xAxisA, probA, color='#ff0000', label='Bleed')
    probA = -1 * (probA - 1)
    axs[r, c].scatter(yAxisA, xAxisA, probA, color='#000000', label='No Bleed')
    probA = -1 * (probA - 1)
    axs[r, c].legend()
    axs[r, c].set_title('Threshold ' + str(i))

    axs[r + 1, c].scatter(yAxis, xAxis, zeroM, color='#000000')
    axs[r + 1, c].scatter(yAxisA, xAxisA, testYA, color='#ff0000', label='Bleed')
    testYA = -1 * (testYA - 1)
    axs[r + 1, c].scatter(yAxisA, xAxisA, testYA, color='#000000', label='No Bleed')
    testYA = -1 * (testYA - 1)
    axs[r + 1, c].legend()
    axs[r + 1, c].set_title('Image ' + str(j) + ' True')

    c = c + 1

    axs[r, c].scatter(yAxis, xAxis, zeroM, color='#000000')
    axs[r, c].scatter(yAxisB, xAxisB, probB, color='#ff0000', label='Bleed')
    probB = -1 * (probB - 1)
    axs[r, c].scatter(yAxisB, xAxisB, probB, color='#000000', label='No Bleed')
    probB = -1 * (probB - 1)
    axs[r, c].legend()
    axs[r, c].set_title('Threshold ' + str(i))

    axs[r + 1, c].scatter(yAxis, xAxis, zeroM, color='#000000')
    axs[r + 1, c].scatter(yAxisB, xAxisB, testYB, color='#ff0000', label='Bleed')
    testYB = -1 * (testYB - 1)
    axs[r + 1, c].scatter(yAxisB, xAxisB, testYB, color='#000000', label='No Bleed')
    testYB = -1 * (testYB - 1)
    axs[r + 1, c].legend()
    axs[r + 1, c].set_title('Image ' + str(k) + ' True')
    fig.set_figheight(12)
    fig.set_figwidth(15)

    c = c + 1
    if c == 4:
        c = 0
        r = r + 2
    i = i / 2
# Below here is code for ROC curves
# predict probabilities
plt.tight_layout()
plt.savefig('Thresholds.jpg')
plt.show()
lr_probA = model.predict_proba(testXA)

# keep probabilities for the positive outcome only
lr_probA = lr_probA[:, 1]

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testYA, ns_probA)
lr_fpr, lr_tpr, _ = roc_curve(testYA, lr_probA)
# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, label='Logistic')
# end for loop here

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()

# show the plot
plt.show()

