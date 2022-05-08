import random
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

data = pd.read_excel('../data/CORK STOPPERS.xls', sheet_name='Data')
data = data.iloc[1:, :]

train, test = train_test_split(data, test_size=0.3)
xtrain = train.drop(labels=['no', 'class'], axis=1)
ytrain = train['class']

xtest = test.drop(labels=['no', 'class'], axis=1)
ytest = test['class']

while True:
    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=random.randint(1, 234125145), splitter='random')
    clf = clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print(score)
    if score >= 0.9:
        result = clf.predict(xtest)
        test['predict'] = result
        test.to_excel('result_gini_sklearn.xlsx')
        break

# prune tree
# import matplotlib.pyplot as plt
# test = []
# for i in range(10):
#     clf1 = tree.DecisionTreeClassifier(max_depth=i+1, random_state=204, splitter='random')
#     clf1 = clf1.fit(xtrain, ytrain)
#     score = clf1.score(xtest, ytest)
#     test.append(score)
# plt.plot(range(1, 11), test, color='red')
# plt.xlabel('max depth')
# plt.ylabel('score')
# plt.title('prune curve')
# plt.show()



# draw the tree
# import os
# os.environ['PATH'] = os.pathsep + r'E:\CodeTools\workspace\Graphviz\bin'
# features_name = list(data.columns)[2:]
# dot_data = tree.export_graphviz(clf, feature_names=features_name,class_names=['0','1','2'], filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# graph.render('Gini tree')
