import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class DecisionTree(object):
    def __init__(self):
        self.nodes = 0
        self.trainSet = None
        self.testSet = None
        self.tree = None

    '''
    read, analyse the file, and split the train/test dataset
    '''

    def creatDataSet(self, path):
        dataset = pd.read_excel(path, sheet_name='Data')
        dataset = dataset.iloc[1:, 1:]

        train, test = train_test_split(dataset, test_size=0.3)
        self.trainSet = train
        self.testSet = test
        return self.trainSet

    '''
    cal the dataset entropy
    '''
    def calEnt(self, data):
        n = data.shape[0]
        iset = data['class'].value_counts()
        p = iset / n
        ent = (-p * np.log2(p)).sum()
        return ent

    '''
    calculate the entropy for the dataset split by one feature
    if feature's type is str, feature = [feature]
    elif feature's type is not, feature = [feature, threshold]
    '''
    def calEnt_feature(self, data, feature):
        ent_feature = 0
        if len(feature) > 1:
            '''
            feature is not str, C4.5 use a binary classification
            '''
            threshold = feature[1]
            feature = feature[0]
            data1 = data[data[feature] <= threshold]
            data2 = data[data[feature] > threshold]
            ent_feature += (data1.shape[0] / data.shape[0]) * self.calEnt(data1)
            ent_feature += (data2.shape[0] / data.shape[0]) * self.calEnt(data2)
        else:
            '''
            feature is str, cal each value entropy
            '''
            trainDataSet = set(data[feature])

            for i in trainDataSet:
                ent_feature += data[data[feature] == i].shape[0] / data.shape[0] \
                               * self.calEnt(data[data[feature] == i])
        return ent_feature

    '''
    choose one node by gain ratio
    first, cal ent_dataset
    then, find those features which gain is bigger than mean_gain
    (note: for each feature, if they are continuous value, then find the best split point)
    final, for those features, cal the gain ratio
    so, we can just get the answer
    Isn't it fun and interesting?
    '''
    def selectNodeByGain_ratio(self, dataset):
        # cal dataset entropy
        ent_D = self.calEnt(dataset)
        data = dataset.drop(labels='class', axis=1)
        # each feature's best ent
        ent_features = []
        thresholds = []

        # cal each feature's ent
        for i in range(data.shape[1]):
            values = sorted(data.iloc[:, i].value_counts().index)
            best_ent = 0xffffffff
            threshold = values[0]

            for j in range(len(values) - 1):
                num = (values[j] + values[j + 1]) / 2
                calEnt_num = self.calEnt_feature(dataset, [data.columns[i], num])
                if calEnt_num < best_ent:
                    best_ent = calEnt_num
                    threshold = num

            ent_features.append(best_ent)
            thresholds.append(threshold)

        thresholds = np.array(thresholds)
        ent_features = np.array(ent_features)
        gains = ent_D - ent_features
        mean_gain = np.mean(gains)

        loc = np.where(gains > mean_gain)

        ans = 0
        Gain_ratio = 0
        features = data.columns
        for i in loc[0]:
            IV = 0
            IV -= (data[data[features[i]] <= thresholds[i]].shape[0] / data.shape[0]) * np.log2(
                data[data[features[i]] <= thresholds[i]].shape[0] / data.shape[0])
            IV -= (data[data[features[i]] > thresholds[i]].shape[0] / data.shape[0]) * np.log2(
                data[data[features[i]] > thresholds[i]].shape[0] / data.shape[0])
            if Gain_ratio < (gains[i] / IV):
                Gain_ratio = gains[i] / IV
                ans = i

        return features[ans], thresholds[ans]

    def buildDecisionTree(self, dataSet):
        ans = self.createDecisionTree(dataSet)
        self.tree = ans
        return ans

    '''
    use the recursion to build the decision tree
    and print the info to watch the change of dataset
    '''
    def createDecisionTree(self, dataSet):
        self.nodes += 1
        print('Build node No.', self.nodes)
        print('Size of dataset is ', dataSet.shape[0])
        # features = list(dataSet.columns)
        labels = dataSet['class'].value_counts()

        if len(dataSet) == 0:
            return None

        if labels.shape[0] == 1 or dataSet.shape[0] == 1:
            return labels.index[0]

        # index = [属性，阈值]
        index = self.selectNodeByGain_ratio(dataSet)
        bestFeature = index[0]

        myTree = {bestFeature: {}}

        left_subtree = self.createDecisionTree(
            dataSet[dataSet[bestFeature] <= index[1]].drop(labels=bestFeature, axis=1))
        if left_subtree is not None:
            myTree[bestFeature]['<' + str(index[1])] = left_subtree
        else:
            myTree[bestFeature]['<' + str(index[1])] = labels.sort_index().index[0]

        right_subtree = self.createDecisionTree(
            dataSet[dataSet[bestFeature] > index[1]].drop(labels=bestFeature, axis=1))
        if right_subtree is not None:
            myTree[bestFeature]['>' + str(index[1])] = right_subtree
        else:
            myTree[bestFeature]['>' + str(index[1])] = labels.sort_index().index[0]

        return myTree

    def setTree(self, tree):
        self.tree = tree

    def save_Tree(self, path='tree.npy'):
        np.save(path, self.tree)

    def predict(self):
        ans = []
        for i in range(self.testSet.shape[0]):
            ans.append(self.predict_one(self.testSet.iloc[i, :]))
        temp = self.testSet
        temp['predict'] = ans
        return temp[['class', 'predict']]

    def predict_one(self, one):
        # feature = firstStr = next(iter())
        t = self.tree
        while True:
            if type(t) != dict:
                return t
            feature = list(t.keys())[0]
            num = t[feature]

            split_point = list(num.keys())[0][1:]
            threshold = (float)(split_point)
            if one[feature] <= threshold:
                t = num['<' + split_point]
            else:
                t = num['>' + split_point]

    def acc_predict(self, predictData):
        return predictData[predictData['class'] == predictData['predict']].shape[0] / predictData.shape[0]

    def analyse_result(self, predictData):
        cfm = confusion_matrix(predictData['class'], predictData['predict'])
        print('The confusion matrix')
        print(cfm)
        plt.matshow(cfm, cmap=plt.cm.gray)
        plt.show()

        print('The micro-p is {:.4f}'.format(precision_score(predictData['class'], predictData['predict'], average='micro')))
        print('The micro-r is {:.4f}'.format(recall_score(predictData['class'], predictData['predict'], average='micro')))
        print('The micro-f1 is {:.4f}'.format(f1_score(predictData['class'], predictData['predict'], average='micro')))


decisionTree = DecisionTree()
trainDataset = decisionTree.creatDataSet('../data/CORK STOPPERS.xls')
print(trainDataset.head())
tree = decisionTree.buildDecisionTree(trainDataset)
print(tree)
# predict testData's label
predictData = decisionTree.predict()
print(predictData)
# cal the acc
print(decisionTree.acc_predict(predictData))
# if you want to save it, just call save, default path has been set
# decisionTree.save_Tree()

# or we can not build the tree, just load the tree we built before,like the codes below
# tree = np.load('tree.npy', allow_pickle=True).item()
# decisionTree.setTree(tree)
# to analyse the matrix, we can save the predictData
# predictData.to_excel('result_c4_5.xlsx')
