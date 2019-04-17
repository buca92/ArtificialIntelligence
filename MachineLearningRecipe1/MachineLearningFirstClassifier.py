from sklearn import tree

# [["Weight", "1 = Smooth, 0 = Bumpy"]]
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 = Apple, 1 = Orange
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[150, 0]]))