import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="User Churn Predictor", layout="wide")

st.title("📉 유저 이탈 예측 대시보드")

# 샘플 데이터 생성
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'login_count': np.random.poisson(10, 200),
        'visit_days': np.random.randint(5, 30, 200),
        'purchases': np.random.poisson(2, 200),
        'support_calls': np.random.binomial(1, 0.2, 200),
        'churn': np.random.binomial(1, 0.3, 200)
    })
    return data

data = load_data()
st.subheader("📊 유저 행동 데이터")
st.dataframe(data.head())

# 훈련 및 예측
X = data.drop("churn", axis=1)
y = data["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 성능 리포트
report = classification_report(y_test, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.subheader("📈 모델 성능 요약")
st.dataframe(report_df.style.background_gradient(cmap='Blues'))

# 사용자 입력 예측
st.subheader("👤 사용자 행동 예측 시뮬레이션")
login = st.slider("로그인 횟수", 0, 50, 10)
visits = st.slider("방문일수", 1, 30, 10)
purchase = st.slider("구매 횟수", 0, 10, 2)
support = st.selectbox("고객센터 이용 여부", [0, 1], format_func=lambda x: "이용 안함" if x==0 else "이용함")

user_input = pd.DataFrame({
    'login_count': [login],
    'visit_days': [visits],
    'purchases': [purchase],
    'support_calls': [support]
})

if st.button("이탈 예측하기"):
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][prediction]
    if prediction == 1:
        st.error(f"❌ 이 유저는 이탈할 가능성이 높습니다. (예측 확률: {prob:.2f})")
    else:
        st.success(f"✅ 이 유저는 잔존할 가능성이 높습니다. (예측 확률: {prob:.2f})")
