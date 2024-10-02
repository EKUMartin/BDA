# 필수과제1 (타이타닉데이터셋)
# - VarianceThreshold -타이타닉 데이터 feature_selection
# - 임계값 기준을 몇으로 했는지? 0.05
# - 그 기준의 이유: 시각화를 통해 분산이 큰 feature들이 많이 없다는 걸 확인하고 정했습니다
# - 어떤 식으로 찾았는지!: 0.1,0.05,0.01를 순서대로 넣어서 피처값을 확인하고 타당하다고 생각하는 것으로 선택했습니다. 
# - 어떤 피처가 선택이 되었나? ['pclass', 'age', 'sibsp', 'parch', 'alone']

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

tt = sns.load_dataset('titanic')
tt['age']=tt['age'].fillna(tt['age'].mean())
tt['fare']=tt['fare'].fillna(tt['fare'].mean())
# tt['age'].fillna(tt['age'].mean(), inplace=True)
# tt['fare'].fillna(tt['fare'].mean(), inplace=True)
x=tt.drop(['survived', 'sex','fare','class','who','embarked', 'embark_town','deck','adult_male','alive'],axis=1)
y = tt['survived']
plt.figure(figsize=(10, 8))
sns.boxplot(data=x)
plt.xticks(rotation=90)
plt.show()
##['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare','embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone']
# X=X.drop(['sex'], axis=1)
# onehot_encoder =OneHotEncoder(drop='first')
selector = VarianceThreshold(threshold=0.05)
# x_onehot= onehot_encoder.fit_transform(X)
X_feature=selector.fit_transform(x)
selected_features=x.columns[selector.get_support(indices=True)]
print("피쳐:",selected_features)
plt.figure(figsize=(8,8))
plt.barh(width=selector.variances_, y=x.columns)
plt.show()