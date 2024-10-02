import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np.random.seed(42)
sample= 500

X1 = np.random.rand(sample) * 100  
X2 = np.random.rand(sample) * 100 
X3 = np.random.rand(sample) * 100 
X4 = np.random.rand(sample) * 100 
noi = np.random.rand(sample) * 100 #노이즈

y = 0.1*X1 + 10*X2 +np.random.randn(sample)*10-X4 #x1은 약한, x2는 강한,x4 영향이 있는 관계
df=pd.DataFrame({
    'X1':X1,
    'X2':X2,
    'X3':X3,
    'X4':X4,
    'noise':noi,
    'y':y
})
#상관계수확인
cor=df.corr()
# print(cor)
#              X1        X2        X3        X4     noise         y
# X1     1.000000  0.010354  0.053966 -0.014562  0.004400  0.019903
# X2     0.010354  1.000000 -0.025912  0.000523 -0.027506  0.994551
# X3     0.053966 -0.025912  1.000000  0.005223  0.103376 -0.026354
# X4    -0.014562  0.000523  0.005223  1.000000  0.000256 -0.098155
# noise  0.004400 -0.027506  0.103376  0.000256  1.000000 -0.027853
# y      0.019903  0.994551 -0.026354 -0.098155 -0.027853  1.000000
cory=df.corr()['y'].drop('y')
# print(cory)
# X1       0.019903
# X2       0.994551
# X3      -0.026354
# X4      -0.098155
# noise   -0.027853
#전체 데이터 학습 linear regression
X = df.drop(columns=['y'])
X_train, X_test, y_train, y_test = train_test_split(X,df['y'],test_size=0.5,random_state=42)

#train
model = LinearRegression()
model.fit(X_train, y_train)

#mse
y_pred =model.predict(X_test)
mse =mean_squared_error(y_test, y_pred)
print('전체데이터:',mse)
# 전체데이터: 86.59127125332869

#0.05 이상인 특성만
threshold=0.05
s_features =cory[abs(cory)>threshold].index#상관계수가 0.05이상인 feature들의 인덱스 출력

# print(s_features)
#Index(['X2', 'X4'], dtype='object')

#LinearRegression
#학습용, 테스트용 데이터 구축
X_selected = df[s_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected,df['y'],test_size=0.5,random_state=42)

#train
model = LinearRegression()
model.fit(X_train, y_train)

#필터 mse
y_pred =model.predict(X_test)
mse_selected =mean_squared_error(y_test, y_pred)
print('필터링 데이터:',mse_selected)
#필터링 데이터: 92.25619709130201
