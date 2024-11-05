# 1 - age (numeric)
# 2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
# 3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
# 4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
# 5 - default: has credit in default? (categorical: "no","yes","unknown")
# 6 - housing: has housing loan? (categorical: "no","yes","unknown")
# 7 - loan: has personal loan? (categorical: "no","yes","unknown")

# # Related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: "cellular","telephone")
# 9 - month: last contact month of year (categorical: "jan", "feb", "mar", …, "nov", "dec")
# 10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# 필수과제2(직접 제가 드린 데이터셋)
# - 정말 피처가 많은 데이터
# - 그 데이터를 피처 셀렉션해서 실제 어떤 피처만 추출할지?
# → 기준에 대한 이유
# → 코드(주석 설명)
# → 실제 선택된 피처는 무엇인지?
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest,chi2
#csv파일 불러오기
df=pd.read_csv('./bank-additional.csv',delimiter=';')
# 결측치 확인
print(df.isnull().sum())
#피쳐들 선택 duration은 관계도가 높으므로 제외 그외는 중요도를 구분할 수 없어 추가
df=df[["age","job","marital","education","default","housing","loan","contact","month","day_of_week","y"]]

# str들 int로 대체
for col in df.columns:
    if df[col].dtype == 'object':
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
#x2 수행을 위한 데이터 준비
x=df.drop(columns=['y'],axis=1)
y=df['y']
chi_scores=chi2(x,y)
print(chi_scores)
#(array([3.98011904e+01, 1.00002637e+01, 2.47147732e+00, 2.28092503e+01,
    #    1.95190032e+01, 3.37326897e-03, 1.05018391e+00, 5.00669927e+01,
    #    1.29857083e-01, 1.60440575e-01, 1.82267291e+05]), array([2.81173679e-10, 1.56517810e-03, 1.15928917e-01, 1.78902545e-06,
    #    9.96038637e-06, 9.53685025e-01, 3.05464733e-01, 1.48585473e-12,
    #    7.18580266e-01, 6.88751214e-01, 0.00000000e+00]))
#카이제곱 시각화
chi_values=pd.Series(chi_scores[0],index=x.columns)
chi_values.sort_values(ascending=True,inplace=True)
chi_values.plot.bar(x='features',y='val',rot=0)
plt.show()#상위 5개 contact, age,education,default,job 피쳐 실질적으로 선택
