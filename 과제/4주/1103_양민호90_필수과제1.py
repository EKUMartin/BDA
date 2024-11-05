# - RFECV
# - SFS 
#     - 두 가지 방법론을 가지고 가장 중요한 피처를 선정하는 코드와 주석으로 정리해서 공유 주세요.
#     - 도메인과 함께 어떤 피처를 선택하는 것이 좋겠다 라는 결론을 부탁드립니다.
import pandas as pd
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
df_mat=pd.read_csv('./student-mat.csv',delimiter=';')
df_por=pd.read_csv('./student-por.csv',delimiter=';')
df_mat=pd.get_dummies(df_mat,columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','internet','romantic'], drop_first=True)
df_por=pd.get_dummies(df_por,columns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','internet','romantic'], drop_first=True)
