import numpy as np
from sklearn import linear_model

from utilities import visualize_classifier

greyhounds = 100
labs = 100
sausages = 100

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)
sausage_height = 14 + 4 * np.random.randn(sausages)

grey_weight = 30 + 2 * np.random.randn(greyhounds)
lab_weight = 34 + 2 * np.random.randn(labs)
sausage_weight = 9 + 2 * np.random.randn(sausages)

# [["height", "weight"]]
dogFeature = []
# 0 - greyhound, 1 = lab, 2 = sausage dog
dogTarget = []
for i in range(len(grey_height)):
    dogFeature.append([grey_height[i], grey_weight[i]])
    dogTarget.append(0)

for i in range(len(lab_height)):
    dogFeature.append([lab_height[i], lab_weight[i]])
    dogTarget.append(1)

for i in range(len(sausage_height)):
    dogFeature.append([sausage_height[i], sausage_weight[i]])
    dogTarget.append(2)

X = np.array(dogFeature)
y = np.array(dogTarget)

# Create the logistic regression classifier
classifier = linear_model.LogisticRegression(solver='liblinear', C=1)

# Train the classifier
classifier.fit(X, y)

# Visualize the performance of the classifier
visualize_classifier(classifier, X, y)