import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd


titanic=sns.load_dataset('titanic') 
titanic.dropna(subset=['age','embarked'], inplace=True)
titanic=pd.get_dummies(titanic, columns = ['sex','embarked','class'], drop_first=True)
X =titanic.drop(['survived','deck','alone','who','adult_male','alive','embark_town'], axis =1)
X =X*1
y = titanic['survived']
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=111)
model =LogisticRegression(max_iter=100)
#SequentialFeatureSelector
#SFS (모델을 선정할 때 어떤 방법으로 선정할지, 선정방법, scoring =어떤 평가로 볼건지?, cv= 교차검증값들(folds))
#모델 선정 기준 설정
sfs_for =SFS(model, k_features='best', forward=True, floating= False, scoring='accuracy',cv=3)#default folds=5
sfs_back =SFS(model, k_features='best', forward=False, floating= False, scoring='accuracy',cv=3)
sfs_step =SFS(model, k_features='best', forward=True, floating= True, scoring='accuracy',cv=3)


# 각 방법을 학습하고 확인해 보기 (데이터 입력)
sfs_for_=sfs_for.fit(X_train, y_train)
sfs_back_=sfs_back.fit(X_train, y_train)
sfs_step_=sfs_step.fit(X_train, y_train)
#선택된 특성 출력

print('forward:')
print(sfs_for_.k_feature_names_)

print('backward:')
print(sfs_back_.k_feature_names_)

print('stepwise:')
print(sfs_step_.k_feature_names_)

#시각화를 통해서 피처를 선택했을 때 어떤 식의 평가나오는지

fig, ax= plt.subplots(1,3, figsize=(18,6))
ax[0].plot(range(1, len(sfs_for_.subsets_)+1),[sfs_for_.subsets_[i]['avg_score'] for i in sfs_for_.subsets_],)
ax[0].set_title('forward')

ax[1].plot(range(1, len(sfs_back_.subsets_)+1),[sfs_back_.subsets_[i]['avg_score'] for i in sfs_back_.subsets_],)
ax[1].set_title('backward')

#fig, ax= plt.subplots(1,3, figsize=(18,6))
ax[2].plot(range(1, len(sfs_step_.subsets_)+1),[sfs_step_.subsets_[i]['avg_score'] for i in sfs_step_.subsets_],)
ax[2].set_title('stepwise')

X,y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=100)
selector =RFE(estimator= model, n_features_to_select = 2)
selector =selector.fit(X,y)
print('feature 값:',selector.support_)
print('feature 순위(숫자):',selector.ranking_)
## RFECV
## 교차검증이 필요하다.

model =RandomForestClassifier()
cv = StratifiedKFold(3)

selector = RFECV(estimator= model, step=1, cv=cv)#step: 매 회마다 지울 feature의 수 / min_feature: 최소한의 피쳐수, 스텝으로 안나눠져도 가능
selector=selector.fit(X,y)
print('결과 개수:',selector.n_features_)
print('결과: ',selector.support_)


# 독립변수와 종속변수 설정
X = titanic.drop(['survived','deck','alone','who','adult_male','alive','embark_town'], axis=1) 
y = titanic['survived']

# 훈련 및 테스트 데이터셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

# 모델 정의
models = {
    'LogisticRegression': LogisticRegression(max_iter=100),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(kernel='linear')#kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
}

# 교차 검증 설정
cv = StratifiedKFold(3)

# 결과를 정리할 DataFrame
results = pd.DataFrame(columns=['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'Selected Features'])

# 각 모델에 대해 RFECV 수행 및 결과 저장
for name, model in models.items():
    selector = RFECV(estimator=model, step=1, cv=cv, scoring='accuracy')
    selector.fit(X_train, y_train)
    
    # Train 및 Test 데이터셋에 대한 평가
    for data in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        dataset_name, X_data, y_data = data
        y_pred = selector.predict(X_data)
        
        accuracy = accuracy_score(y_data, y_pred)
        precision = precision_score(y_data, y_pred)
        recall = recall_score(y_data, y_pred)
        f1 = f1_score(y_data, y_pred)
        
        selected_features = ', '.join(X.columns[selector.support_])
        
        # 결과를 DataFrame에 추가
        result_row = pd.DataFrame({
            'Model': [name],
            'Dataset': [dataset_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1],
            'Selected Features': [selected_features]
        })
        results = pd.concat([results, result_row], ignore_index=True)
        
print(results)
