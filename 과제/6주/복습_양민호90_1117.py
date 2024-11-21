import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# #보루타
df = sns.load_dataset('titanic')
df = df.dropna(subset=['age','embarked','deck'])
df['sex'] =df['sex'].map({'male':0,'female':1})
df['embarked'] =df['embarked'].astype('category').cat.codes
df['deck'] = df['deck'].astype('category').cat.codes
X = df[['pclass','sex','age','sibsp','parch','fare','embarked','deck']]
y = df['survived']

## 보루타를 통한 피처셀렉션을 위해 RandomForestClassifier

rf= RandomForestClassifier(class_weight = 'balanced', max_depth=20)# depth:10  ['sex', 'age', 'fare'] /depth:20 ['sex','age']
boruta_selector = BorutaPy(rf, n_estimators = 'auto', random_state=111)
boruta_selector.fit(X.values, y.values) #행렬로 대입해서

print('선택된 특성',X.columns[boruta_selector.support_].tolist())



green_area = X.columns[boruta_selector.support_].tolist()
blue_area = X.columns[boruta_selector.support_weak_].tolist()

feature_importance = boruta_selector.ranking_ #중요도 랭킹

plt.figure(figsize=(15,6))
plt.bar(X.columns, feature_importance,color= 'grey')
plt.bar(green_area, [1]*len(green_area), color='green', label='Seleted')
plt.bar(blue_area, [2]*len(blue_area), color='blue', label='Not Seleted')
plt.legend()


#shap
X= df[['sex','age','fare','pclass']]
y= df['survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=111)
#모델학습
model = RandomForestClassifier(random_state=111)
model.fit(X_train, y_train)
#shap 값 구하기
explainer=shap.TreeExplainer(model)#tree라 2진 분류기-> 클래스 0,1로 구분/ 다중이면 0,1,2....
shap_values=explainer.shap_values(X_test)

# 특정 클래스 선택 (예: 클래스 0)
shap_values_class_0 = shap_values[:, :, 0]#0:sex,fare, age 1:sex,fare,age

# 요약 플롯 생성
feature_names = X_test.columns.tolist()  # X_test가 DataFrame일 경우
fig=plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax=fig.add_subplot()
shap.summary_plot(shap_values_class_0, X_test, plot_type='bar', feature_names=X_test.columns.tolist())
plt.show()
