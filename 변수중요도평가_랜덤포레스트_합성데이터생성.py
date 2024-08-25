import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from synthpop import Synthpop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 현재 작업 디렉토리 확인
current_path = os.getcwd()
print(f"현재 작업 디렉토리: {current_path}")

# 읽어올 엑셀 파일과 경로 지정
filename='C:\\Users\\user\\Downloads\\20240819\\대출가망고객_랜덤포레스트모델분석_원본.csv'

# 엑셀 파일 읽어오기
df = pd.read_csv(filename)
#df = df.drop('Unnamed: 0', axis=1)

# 데이터프레임 출력
print(df)

# 범주형 변수 원핫 인코딩
data = pd.get_dummies(df, drop_first=True)

X = data.drop('loan_needed_Y', axis=1)  # 독립변수
y = data['loan_needed_Y']               # 종속변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 스케일링
# 데이터셋의 각 특성(feature)들이 서로 다른 범위를 가질 때, 
# 모델이 특정 특성에 더 큰 가중치를 부여. 
# 예를 들어, 나이(age)는 0에서 100 사이의 값을 가질 수 있지만, 
# 소득(income)은 수천에서 수십만 단위의 값을 표현, 이러한 차이를 줄이기 위해 스케일링을 통해 모든 특성을 동일한 범위로 처리
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 변수 중요도 추출
importances = model.feature_importances_
feature_names = X.columns

# 중요도 시각화
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_col = importance_df.iloc[:,0].to_list()
importance_col.append('loan_needed_Y')

# 바차트 생성
fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
fig.show()

# 결과 출력
print(importances)

# 예측 및 평가
y_pred = model.predict(X_test)

# 평가 결과 출력
# 혼동 행렬은 모델의 예측 성능을 다양한 지표로 나타내는 데 유용
# [[TN, FP]
#  [FN, TP]]
print('원본')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

##############################################################
# 변수의 중요도 순서대로 합성데이터 생성
##############################################################
from synthpop import Synthpop

# 데이터 타입 지정
data.info()
data1 = data[importance_col]
data1['loan_amount'] = data1['loan_amount'].astype(int)
data1['income'] = data1['income'].astype(int)
data1['credit_score'] = data1['credit_score'].astype(int)
data1['age'] = data1['age'].astype(int)
data1.info()
dtypes_dict = data1.dtypes.apply(lambda x: x.name).to_dict()
print(dtypes_dict)
dtypes_dict = {'loan_amount': 'int', 'income': 'int', 'credit_score': 'int', 'age': 'int', 'loan_purpose_주택자금': 'bool', 'gender_Male': 'bool', 'loan_purpose_자동차구입': 'bool', 'loan_purpose_학자금대출': 'bool', 'loan_purpose_일반대출': 'bool', 'loan_needed_Y': 'bool'}

# Synthpop 모델 초기화 및 학습
spop = Synthpop()
spop.fit(data1, dtypes_dict)

# 합성 데이터 생성
synth_df = spop.generate(len(data1))
print(synth_df.head())


##############################################################
# 랜덤포레스트 모델 학습 후 모델성능 분석 하기
##############################################################

data = synth_df

X = data.drop('loan_needed_Y', axis=1)  # 독립변수
y = data['loan_needed_Y']               # 종속변수

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 스케일링
# 데이터셋의 각 특성(feature)들이 서로 다른 범위를 가질 때, 
# 모델이 특정 특성에 더 큰 가중치를 부여. 
# 예를 들어, 나이(age)는 0에서 100 사이의 값을 가질 수 있지만, 
# 소득(income)은 수천에서 수십만 단위의 값을 표현, 이러한 차이를 줄이기 위해 스케일링을 통해 모든 특성을 동일한 범위로 처리
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 변수 중요도 추출
importances = model.feature_importances_
feature_names = X.columns

# 중요도 시각화
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_col = importance_df.iloc[:,0].to_list()
importance_col.append('loan_needed_Y')

# 바차트 생성
fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importances')
fig.show()

# 결과 출력
print(importances)

# 예측 및 평가
y_pred = model.predict(X_test)

# 평가 결과 출력
# 혼동 행렬은 모델의 예측 성능을 다양한 지표로 나타내는 데 유용
# [[TN, FP]
#  [FN, TP]]
print('합성데이터')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
