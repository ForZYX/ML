import pandas as pd
import numpy as np
from fastai import *
from fastai.tabular.all import *
import seaborn as sns
import matplotlib.pyplot as plt

# 忽略警告
pd.options.mode.chained_assignment = None
# 设置输出列数，防止折叠输出
pd.options.display.max_columns = 999

# test shape: (418, 11)
test = pd.read_csv('../data/Titanic/test.csv')
# train shape: (891, 12)
train = pd.read_csv('../data/Titanic/train.csv')

# to_frame: convert Series to DataFrame
# 统计每一个属性的缺失值
print("Null distribution")
print(train.isnull().sum().sort_values(ascending=False).to_frame().T)
'''
    Cabin  Age  Embarked  PassengerId  Survived  Pclass  Name  Sex  SibSp
      687  177         2            0         0       0     0    0      0
    Parch  Ticket  Fare  
        0       0     0 
'''

# 读取拓展数据
# test extended shape: (418, 20)
test_ext = pd.read_csv('../data/Titanic/extendedData/test.csv')
# train extended shape: (891, 21)
train_ext = pd.read_csv('../data/Titanic/extendedData/train.csv')
# 统计每一个属性的缺失值
print("Null distribution")
print(train_ext.isnull().sum().sort_values(ascending=False).to_frame().T)
'''
   Body  Cabin  Lifeboat  Age  Age_wiki  Embarked  Destination  Boarded  \
    804   687       546  177         4         2            2        2   
   Hometown  Name_wiki  WikiId  Class  Survived  Fare  Ticket  Parch  SibSp  \
          2          2       2      2         0     0       0      0      0   
   Sex  Name  Pclass  PassengerId  
     0     0       0            0  
'''

print('# Uniques')
print(train_ext.nunique().sort_values(ascending=False).to_frame().T)
'''
   PassengerId  Name  Name_wiki  WikiId  Ticket  Hometown  Fare  Destination  \
           891   891        889     889     681       437   248          234   
   Cabin  Age  Body  Age_wiki  Lifeboat  Parch  SibSp  Boarded  Embarked  \
     147   88    87        74        22      7      7        4         3   
   Pclass  Class  Survived  Sex  
        3      3         2    2  
'''

# 去除不需要的数据
req_cols = train_ext.columns.difference(['Body', 'Cabin' # high null of features
                            , 'Age', 'Class', 'Name', 'Embarked' # duplicate features
                            , 'PassengerId', 'Name_wiki', 'WikiId'] # high cardinality features
                            ).tolist()

print('Required features: ', req_cols)

# 获得最终数据集
train_ext_filter = train_ext[req_cols]
test_ext_filter = test_ext[[col for col in req_cols if col != 'Survived']]

# 分离数据
# categorical features
cat_feat = train_ext_filter.select_dtypes('object').columns.tolist()

# continuous features
cont_feat = train_ext_filter.columns.difference(cat_feat + ['Survived']).tolist()

print('categorical data:\n', train_ext_filter[cat_feat].head())

print('continuous data:\n', train_ext_filter[cont_feat].head())

# 划分验证集
splits = RandomSplitter(valid_pct=0.2, seed=20020204)\
                        (range_of(train_ext_filter))

# 创建TabularPandas
tp = TabularPandas(train_ext_filter,
                   procs=[Categorify, FillMissing, Normalize],
                   cat_names=cat_feat,
                   cont_names=cont_feat,
                   y_names='Survived',
                   y_block=CategoryBlock(),
                   splits=splits)

train_dl = TabDataLoader(tp.train, bs=64, shuffle=True, drop_last=True)
val_dl=TabDataLoader(tp.valid, bs=64)

dls = DataLoaders(train_dl, val_dl)

learn = tabular_learner(dls, layers=[30,10], metrics=accuracy)

# 查看网络结构
# learn.summary()
# 寻找最大学习率
# learn.lr_find()

learn.fit_one_cycle(10, lr_max=0.0036)

# plt.plot(range(len(learn.recorder.lrs)), learn.recorder.lrs)

# plt.plot(range(len(learn.recorder.losses)),learn.recorder.losses)

# 因为训练数据中fare没有缺失的，无法对测试集进行处理，因此需要提前补齐
fare_med_val = test_ext_filter.Fare.median()
test_ext_filter.loc[:, 'Fare'].fillna(fare_med_val, inplace=True)
t1 = learn.dls.train_ds.new(test_ext_filter)
t1.process()
d1 = TabDataLoader(t1)
pred = learn.get_preds(dl=d1)[0].argmax(1).numpy()

# 输出最终结果
out = pd.DataFrame({'PassengerId': test_ext.PassengerId, 'Survived':pred.astype(int)})
out.to_csv('submission__1.csv', index=False)
# out.head()


