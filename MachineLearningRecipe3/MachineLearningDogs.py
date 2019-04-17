import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500
sausages = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)
sausage_height = 14 + 4 * np.random.randn(sausages)

grey_weight = 30 + 2 * np.random.randn(greyhounds)
lab_weight = 34 + 2 * np.random.randn(labs)
sausage_weight = 9 + 2 * np.random.randn(sausages)

# plt.hist([grey_height, lab_height, sausage_height], stacked=True, color=['r', 'b', 'g'])
# plt.show()

# plt.hist([grey_weight, lab_weight, sausage_weight], stacked=True, color=['r', 'b', 'g'])
# plt.show()

dogFeatureNames = ["Height (inch)", "Weight (kg)"]
dogTargetNames = ["Greyhound", "Labrador", "Sausage dog"]

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

# Greyhound
# testSample = [[24, 20]]
# Labrador
# testSample = [[24, 34]]
# Sausage Dog
testSample = [[24, 15]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(dogFeature, dogTarget)

resultClf = clf.predict(testSample)

print(resultClf)
print(dogTargetNames[resultClf[0]])

# viz code chart
import graphviz
dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=dogFeatureNames,
                                class_names=dogTargetNames,
                                filled=True,
                                rounded=True,
                                impurity=False,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("dogs_classification")

# regression
rgrs1 = tree.DecisionTreeRegressor(max_depth=2)
rgrs1 = rgrs1.fit(dogFeature, dogTarget)

rgrs2 = tree.DecisionTreeRegressor(max_depth=5)
rgrs2 = rgrs2.fit(dogFeature, dogTarget)

resultRgrs1 = rgrs1.predict(testSample)
resultRgrs2 = rgrs2.predict(testSample)

print(resultRgrs1)
print(resultRgrs2)

# plt.figure()
# plt.scatter(dogFeature, dogTarget, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(testSample, resultRgrs1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(testSample, resultRgrs2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()


