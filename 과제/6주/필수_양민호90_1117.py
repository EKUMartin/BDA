# ## 필수과제1
# - 모델을 단순하게 RandomForest만 학습
# - 타이타닉 데이터를 RF모델을 튜닝하여 과적합을 최소화 하여 하이퍼 파라미터를 찾고, 해당 하이퍼파라미터를 가지고 다시 한 번 샤플리를 진행해서
# - 기존 베이스 모델의 샤플리값과 과적합을 최소화한 모델의 하이퍼파라미터로 샤플리값을 추출하는 것
# - 둘을 비교해 주시면 됩니다.
#     - **필수적으로**
#     - 과적합을 최소화했다는 기준
#         - 성능?
#         - 그리드 서치 등을 통해 찾은 것인지?
#         - 시각화 등으로도 보여주면서 과적합을 최소화 했다는 것을 코드와 함께 설명해 주셔야 합니다.

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from boruta import BorutaPy
import shap
from feature_engine.selection import DropCorrelatedFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
#과적합 해결 X
# 데이터 로드
df = sns.load_dataset('titanic')

# 결측값 처리
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 불필요한 열 제거
df.drop(columns=['deck', 'embark_town', 'alive', 'adult_male', 'who', 'class'], inplace=True)

# 범주형 데이터 인코딩
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 독립 변수와 종속 변수 분리
X = df.drop(columns=['survived'])
y = df['survived']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 모델 학습
model = RandomForestClassifier(random_state=10)
model.fit(X_train, y_train)
explainer=shap.TreeExplainer(model)
expected_value = explainer.expected_value ## Base SHAP Value
shap_values_notuning=explainer.shap_values(X_test)
print("예측값:", expected_value)
print("샵합:", np.sum(shap_values_notuning)+expected_value)
shap_values_notuning_class_0 = shap_values_notuning[:, :, 0]
fig=plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax=fig.add_subplot()
shap.summary_plot(shap_values_notuning_class_0, X_test, plot_type='bar', feature_names=X_test.columns.tolist())
plt.show()
# 결과
# 예측값: [0.60728933 0.39271067]
# 샵합: [0.60728933 0.39271067]
# 훈련 정확도: 0.9831
# 테스트 정확도: 0.8268


# 정확도 확인
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"훈련 정확도: {train_accuracy:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")

#과적합 해결 O
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 모델 학습
model = RandomForestClassifier(random_state=10)
#하이퍼 파라미터 튜닝 

#1차로 parameter 후보 수 줄이기
param = {
    'n_estimators': [30,40,50, 60, 70, 80, 90, 100],
    'max_depth': [5, 10,15, 20,25, 30, 35,50],
    'min_samples_split': [2,3,4,5,6,7,8],
    'min_samples_leaf': [1, 2,3, 4,5],
    'max_features': ['auto', 'sqrt', 'log2']
}
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param,
    n_iter=50,  # 시도할 조합 수
    scoring='accuracy',  # 정확도 기준
    cv=5,  # 교차 검증 folds
    verbose=1,
    random_state=42,
    n_jobs=-1  # 병렬 처리
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_
print("최적 하이퍼파라미터:", random_search.best_params_)
#하이퍼파라미터:{'n_estimators': 40, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 15}
#grid search로 최적 후보 선정
param_grid = {
    'n_estimators': [35, 40],
    'max_depth': [12,15],
    'min_samples_split': [5],
    'min_samples_leaf': [4,6],
    'max_features': ['log2']
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("그리드:" , best_rf)
#shap 계산: 최적화param으로 model 생성
model = RandomForestClassifier(max_depth=15, max_features='log2', min_samples_leaf=4,min_samples_split=5, n_estimators=40,random_state=10)
model.fit(X_train, y_train)
explainer=shap.TreeExplainer(model)
expected_value = explainer.expected_value ## Base SHAP Value
shap_values=explainer.shap_values(X_test)
print("예측값:", expected_value)
print("샵합:", np.sum(shap_values)+expected_value)
shap_values_class_0 = shap_values[:, :, 0]
fig=plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax=fig.add_subplot()
shap.summary_plot(shap_values_class_0, X_test, plot_type='bar', feature_names=X_test.columns.tolist())
plt.show()

# 정확도 확인
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"훈련 정확도: {train_accuracy:.4f}")
print(f"테스트 정확도: {test_accuracy:.4f}")


#learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Score")
plt.plot(train_sizes, test_mean, label="Cross-Validation Score")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

#결과 비교
#튜닝전
# 예측값: [0.60728933 0.39271067]
# 샵합: [0.60728933 0.39271067]
# 훈련 정확도: 0.9831
# 테스트 정확도: 0.8268
#튜닝후
# 예측값: [0.60898876 0.39101124]
# 샵합: [0.60898876 0.39101124]
# 훈련 정확도: 0.8764
# 테스트 정확도: 0.8436
#결과:
#훈련정확도가 낮아졌다는 의미는 과적합을 완화했다는 의미입니다. 테스트 정확도가 향상했습니다. SHAP은 차이가 없는걸로 보아 두 모델 모두 주요 피처가 비슷한걸로 해석됩니다.