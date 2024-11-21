# - RFECV
# - SFS 
#     - 두 가지 방법론을 가지고 가장 중요한 피처를 선정하는 코드와 주석으로 정리해서 공유 주세요.
#     - 도메인과 함께 어떤 피처를 선택하는 것이 좋겠다 라는 결론을 부탁드립니다.
import pandas as pd
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from warnings import simplefilter
from sklearn.preprocessing import StandardScaler
simplefilter(action='ignore')
df_mat=pd.read_csv('./student-mat.csv',delimiter=';')
df_por=pd.read_csv('./student-por.csv',delimiter=';')
#data preprocessing
df_mat=pd.get_dummies(df_mat,columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'], drop_first=True)
df_por=pd.get_dummies(df_por,columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'], drop_first=True)

#train, test 데이터 생성
x_por=df_por.drop('G3', axis=1)
y_por=df_por['G3']
X_train, X_test, y_train, y_test = train_test_split(x_por, y_por, test_size=0.2, random_state=111)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("a")
model = LogisticRegression(max_iter=200)
stratified_k = StratifiedKFold(n_splits=2, shuffle=True, random_state=111)
#forward
print("b")
sfs_for = SFS(model, k_features=1, forward=True, floating=False, scoring='accuracy', cv=stratified_k)
sfs_for.fit(X_train_scaled, y_train)
print("c")
#backward
sfs_back = SFS(model, k_features=1, forward=False, floating=False, scoring='accuracy', cv=stratified_k)
sfs_back.fit(X_train_scaled, y_train)
print("c")
#step
sfs_step = SFS(model, k_features=1, forward=True, floating=True, scoring='accuracy', cv=stratified_k)
sfs_step.fit(X_train_scaled, y_train)
print("c")
# 결과
print("Forward Selection:", sfs_for.k_feature_names_)
print("Backward Selection:", sfs_back.k_feature_names_)
print("Stepwise Selection:", sfs_step.k_feature_names_)
#rfecv
rf=RandomForestClassifier(random_state=111)
rfecv = RFECV(estimator=model, step=1, cv=stratified_k, scoring='accuracy')
rfecv_rf = RFECV(estimator=rf, step=1, cv=stratified_k, scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)
rfecv_rf.fit(X_train_scaled, y_train)
print('결과: ',rfecv.get_support(indices=True))
print('결과(rf): ',rfecv_rf.get_support(indices=True))

#Forward Selection: ('14',) studytime
# Backward Selection: ('14',) studytime
# Stepwise Selection: ('14',) studytime
# 결과:[12 13 14]
# 결과(rf):14
x_mat=df_mat.drop('G3', axis=1)
y_mat=df_mat['G3']
X_train, X_test, y_train, y_test = train_test_split(x_mat, y_mat, test_size=0.2, random_state=111)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=200)
stratified_k = StratifiedKFold(n_splits=2, shuffle=True, random_state=111)
#forward
sfs_for = SFS(model, k_features=1, forward=True, floating=False, scoring='accuracy', cv=stratified_k)
sfs_for.fit(X_train_scaled, y_train)
#backward
sfs_back = SFS(model, k_features=1, forward=False, floating=False, scoring='accuracy', cv=stratified_k)
sfs_back.fit(X_train_scaled, y_train)
#step
sfs_step = SFS(model, k_features=1, forward=True, floating=True, scoring='accuracy', cv=stratified_k)
sfs_step.fit(X_train_scaled, y_train)
# 결과
print("Forward Selection:", sfs_for.k_feature_names_)
print("Backward Selection:", sfs_back.k_feature_names_)
print("Stepwise Selection:", sfs_step.k_feature_names_)

#rfecv
rf=RandomForestClassifier(random_state=111)
rfecv = RFECV(estimator=model, step=1, cv=stratified_k, scoring='accuracy')
rfecv_rf = RFECV(estimator=rf, step=1, cv=stratified_k, scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)
rfecv_rf.fit(X_train_scaled, y_train)
print('결과: ',rfecv.get_support(indices=True))
print('결과(rf): ',rfecv_rf.get_support(indices=True))
# Forward Selection: ('14',)
# Backward Selection: ('14',)
# Stepwise Selection: ('14',)
# 결과:[12 13 14]
# 결과(rf):14
#전체 데이터를 기준으론 당연히 공부시간이 곧 성적이기 때문에 studytime이 가장 중요하다고 생각합니다. 하지만 다른 피쳐들도 고려해야 한다면 G1, G2의 성적, 그리고 실패횟수도 
#학생의 G3 성적과 연관되어 있을거라 생각합니다.
