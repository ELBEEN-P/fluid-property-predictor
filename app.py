import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier

# --- [1] 페이지 기본 설정 ---
st.set_page_config(page_title="선박 복원성 AI 예측", page_icon="🌊", layout="wide")

# --- [2] 물리적 특성 공학 (Feature Engineering) ---
def apply_feature_engineering(df):
    X = df.copy()
    X['Draft'] = X['H'] * X['SG']
    X['KB_hint'] = X['Draft'] / 2.0
    X['BM_hint'] = (X['B']**2) / (12.0 * X['Draft'] + 1e-5)
    return X

# --- [3] AI 모델 학습 및 캐싱 ---
@st.cache_resource
def load_and_train_models():
    try:
        df = pd.read_csv('ship_data.csv')
    except FileNotFoundError:
        return None, None, None, None, "오류: 'ship_data.csv' 파일이 없습니다. task_2.py를 먼저 실행하세요."
        
    if len(df) < 4500:
        return None, None, None, None, "오류: 데이터가 5000개 미만입니다. task_2.py를 다시 실행하세요."

    X_base = df[['Shape', 'B', 'H', 'SG', 'KG']]
    X_engineered = apply_feature_engineering(X_base)
    X_encoded = pd.get_dummies(X_engineered, columns=['Shape'])
    feature_cols = X_encoded.columns

    y_reg = df['GM']
    y_clf = df['Status']

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(X_encoded, y_reg, y_clf, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    reg_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=3000, random_state=42)
    reg_model.fit(X_train_s, yr_train)

    clf_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    clf_model.fit(X_train_s, yc_train)

    return reg_model, clf_model, scaler, feature_cols, "Success"

# --- [4] 메인 웹 UI ---
st.title("🌊 AI 기반 선박 복원성 실시간 예측 대시보드")
st.markdown("**White 유체역학 P2.128 이론 검증 및 물리적 특성 공학(Feature Engineering) 적용 모델**")
st.markdown("---")

with st.spinner("AI 모델을 초기화하고 있습니다..."):
    reg_model, clf_model, scaler, feature_cols, status_msg = load_and_train_models()

if status_msg != "Success":
    st.error(status_msg)
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ 선박 및 적재 조건 입력")
    st.info("슬라이더를 조작하면 우측 결과가 실시간으로 변합니다.")
    
    user_shape = st.selectbox("단면 형상 (Shape)", ["Rectangle", "Triangle", "Semicircle"], index=0)
    user_B = st.number_input("선폭 B (m)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
    user_H = st.number_input("선고 H (m)", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
    user_SG = st.slider("비중 SG (밀도비)", min_value=0.01, max_value=0.99, value=0.88, step=0.01)
    user_KG = st.number_input("무게중심 높이 KG (m)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

# --- [5] 실시간 데이터 처리 및 예측 ---
input_df = pd.DataFrame([[user_shape, user_B, user_H, user_SG, user_KG]], 
                        columns=['Shape', 'B', 'H', 'SG', 'KG'])
input_engineered = apply_feature_engineering(input_df)
input_encoded = pd.get_dummies(input_engineered, columns=['Shape'])

for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
        
input_encoded = input_encoded[feature_cols]
input_scaled = scaler.transform(input_encoded)

predicted_gm = reg_model.predict(input_scaled)[0]
predicted_status = clf_model.predict(input_scaled)[0]
status_str = "Stable" if predicted_status == 1 else "Unstable"

# --- [6] 우측 결과 출력 화면 ---
with col2:
    st.subheader("📊 AI 실시간 예측 결과")
    
    st.markdown(f"**AI가 산출한 물리적 추정치:**")
    st.markdown(f"- 흘수(Draft): `{input_engineered['Draft'][0]:.2f} m`")
    st.markdown(f"- 부심 높이(KB) 근사치: `{input_engineered['KB_hint'][0]:.2f} m`")
    st.markdown("---")
    
    st.metric(label="예측된 메타센터 높이 (GM)", value=f"{predicted_gm:.4f} m")
    
    if predicted_status == 1:
        st.success("✅ **예측된 선박 상태: 안정 (Stable)**")
    else:
        st.error("🚨 **예측된 선박 상태: 전복 위험 (Unstable)**")

# --- [7] 데이터 추출 (Download) 기능 추가 ---
st.markdown("---")
st.subheader("💾 분석 결과 내보내기")

# 다운로드용 데이터프레임 생성 (입력값 + 물리힌트 + AI 예측결과 통합)
export_df = input_engineered.copy()
export_df['Predicted_GM (m)'] = predicted_gm
export_df['Status'] = status_str

# 한글 깨짐 방지를 위해 utf-8-sig로 인코딩합니다.
csv_data = export_df.to_csv(index=False).encode('utf-8-sig')

st.download_button(
    label="📥 현재 시나리오 데이터 추출 (CSV)",
    data=csv_data,
    file_name=f"ship_stability_result_{user_shape}.csv",
    mime="text/csv",
    help="현재 화면에 입력된 조건과 AI의 예측 결과를 엑셀에서 열어볼 수 있는 CSV 파일로 다운로드합니다."
)