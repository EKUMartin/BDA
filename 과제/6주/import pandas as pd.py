import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# 데이터 준비
df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'embarked', 'deck'])
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].astype('category').cat.codes
df['deck'] = df['deck'].astype('category').cat.codes

X = df[['sex', 'age', 'fare', 'pclass']]
y = df['survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

# Random Forest 모델 학습
model = RandomForestClassifier(random_state=111)
model.fit(X_train, y_train)

# SHAP Explainer 생성 및 SHAP 값 계산
explainer = shap.TreeExplainer(model)

# SHAP 값을 예측
shap_values = explainer.shap_values(X_test)

# 클래스 1 (생존) 기준으로 요약 플롯 생성
shap.summary_plot(shap_values[1], X_test, plot_type='bar')
