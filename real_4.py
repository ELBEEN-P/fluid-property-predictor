import streamlit as st
import pandas as pd
import joblib
import os

# --- 지수 표기법 변환 및 HTML 카드 UI 생성 함수 ---
def get_html_display(name, value):
    # 소수점 4자리 지수 형태 생성
    s = f"{value:.4e}"
    base, exp = s.split('e')
    exp_int = int(exp)
    
    # 지수가 0이면 일반 소수점, 아니면 10^n (HTML sup 태그) 적용
    val_str = f"{float(base):.4f}" if exp_int == 0 else f"{float(base):.4f} × 10<sup>{exp_int}</sup>"
    
    # 숫자 잘림을 원천 차단하는 깔끔한 카드 스타일
    return f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 6px solid #007bff; margin-bottom: 15px; width: 100%;">
        <p style="margin: 0; font-size: 16px; color: #666; font-weight: bold;">{name}</p>
        <p style="margin: 5px 0 0 0; font-size: 32px; color: #333; font-family: 'Courier New', monospace; white-space: nowrap;">
            {val_str}
        </p>
    </div>
    """

# --- 페이지 기본 설정 (넓게 쓰기) ---
st.set_page_config(page_title="유체 물성 예측 AI", layout="wide")

st.title("💧 AI 기반 유체 물성 예측 모델")
st.markdown("다양한 온도 조건에서 유체의 점성계수, 밀도, 표면장력 등을 즉시 예측합니다.")
st.write("---")

# --- 입력 섹션 ---
col1, col2, col3 = st.columns(3)
with col1:
    # 사용자 선택은 대문자로, 내부 처리용 키는 소문자로 매핑
    fluid_choice = st.selectbox("유체 종류 선택", ["Water", "N2", "Air"])
with col2:
    unit_choice = st.selectbox("온도 단위 선택", ["Kelvin (K)", "Celsius (°C)", "Fahrenheit (°F)"])
with col3:
    input_temp = st.number_input("온도 입력", value=300.0, step=0.1, format="%.2f")

# 온도 단위 변환 함수 (모든 연산은 Kelvin 기준)
def to_kelvin(t, u):
    if "Celsius" in u: return t + 273.15
    if "Fahrenheit" in u: return (t - 32) * 5/9 + 273.15
    return t

temp_k = to_kelvin(input_temp, unit_choice)
st.info(f"💡 모델 분석 온도: **{temp_k:.2f} K**")

# --- 예측 실행 및 결과 출력 ---
if st.button("🚀 물성치 예측 실행", use_container_width=True):
    # 선택된 유체의 모델 파일 경로 확인
    model_path = f"models/{fluid_choice.lower()}_models.pkl"
    
    if os.path.exists(model_path):
        # 학습된 모델 딕셔너리 로드
        fluid_models = joblib.load(model_path)
        input_df = pd.DataFrame({'Temperature (K)': [temp_k]})
        
        st.success(f"### 📊 {fluid_choice} 예측 결과")
        
        # 모델 딕셔너리를 순회하며 개별 예측 수행
        for target_name, model in fluid_models.items():
            prediction = model.predict(input_df)[0]
            # 만들어둔 HTML 함수로 출력
            st.markdown(get_html_display(target_name, prediction), unsafe_allow_html=True)
            
    else:
        st.error("⚠️ 학습된 모델 파일을 찾을 수 없습니다. 터미널에서 모델 학습 코드를 먼저 실행해 주세요.")