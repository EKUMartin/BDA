from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
X = [[0,2,0,3],
    [0,1,2,3],
    [0,1,1,5]]
#임계값을 낮게
selector =VarianceThreshold(threshold=0.01)
X_low_variance =selector.fit_transform(X)
print(X_low_variance)
# [[2 0 3]
#  [1 2 3]
#  [1 1 5]]
#임계값을 높게
selector =VarianceThreshold(threshold=0.5)
X_high_variance =selector.fit_transform(X)
print(X_high_variance)
# [[0 3]
#  [2 3]
#  [1 5]]
#x2
X = np.array([[4,1,4],
             [4,5,5],
             [6,2,3],
             [1,10,4]])
y = np.array([1,1,0,1])
selector=SelectKBest(chi2, k=1)#상위 1개 피쳐만 선택
X_sel=selector.fit_transform(X,y)
print(X_sel)
# [[ 1]
#  [ 5]
#  [ 2]
#  [10]]

#타이타닉
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
tt = sns.load_dataset('titanic')
tt['age'].fillna(tt['age'].mean(), inplace=True)#결측치 평균으로 대체
tt['embark_town'].fillna(tt['embark_town'].mode()[0], inplace=True)
tt['fare'].fillna(tt['fare'].mean(), inplace=True)
# print(tt.columns)
# ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
#        'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
#        'alive', 'alone']
X = tt[['pclass','sex','age','fare','embark_town','deck','alone']] # 사용할 피쳐선택
y = tt['survived']
#분위로 나누기
X.loc[:, 'age_binned']=pd.qcut(X['age'], q=4, labels=False)
X.loc[:, 'fare_binned']=pd.qcut(X['fare'], q=4, labels=False)
#one-hot
X=X.drop(['age','fare'], axis=1)
onehot_encoder =OneHotEncoder(sparse_output=False, drop='first')
X_one = onehot_encoder.fit_transform(X)
#전체에 대한 카이제곱 검정
chi_all =SelectKBest(chi2, k='all')
X_selected_all =chi_all.fit_transform(X_one,y)
chi_all_scores =pd.DataFrame({'Feature': onehot_encoder.get_feature_names_out(X.columns),
    'Score':chi_all.scores_}).sort_values(by='Score', ascending=True)
print(chi_all_scores) 
#상위 5개에 대한 x2
chi_five =SelectKBest(chi2, k=5)
X_selected =chi_five.fit_transform(X_one,y)
#피쳐들 점수
selected_indices=chi_five.get_support(indices=True)#선택된 피쳐들
selected_features=onehot_encoder.get_feature_names_out(X.columns)[selected_indices]#피쳐들 이름
selected_scores = chi_five.scores_[selected_indices]#점수
##간단한 시각화
plt.figure(figsize=(8,4))
plt.barh(selected_features, selected_scores, color='lightblue')
plt.show()