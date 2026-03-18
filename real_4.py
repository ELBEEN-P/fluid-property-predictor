import streamlit as st # 웹 페이지를 만들기 위한 Streamlit 라이브러리를 불러옵니다.
import pandas as pd # 데이터를 표 형태로 다루기 위해 pandas 라이브러리를 불러옵니다.
import joblib # 저장된 인공지능 모델 파일(.pkl)을 읽어오기 위해 joblib을 불러옵니다.
import os # 파일이 폴더에 실제로 존재하는지 확인하기 위해 os 모듈을 불러옵니다.

# --- 유체별 임계 온도 및 삼중점 온도 정의 (기본값: Kelvin) ---
CRITICAL_TEMPS = { # 유체별 임계 온도를 절대온도(K) 기준으로 딕셔너리에 저장합니다.
    "Water": 647.10, # 물의 임계 온도 (K)
    "N2": 126.19 # 질소의 임계 온도 (K)
}
TRIPLE_TEMPS = { # 유체별 삼중점 온도를 절대온도(K) 기준으로 딕셔너리에 저장합니다.
    "Water": 273.16, # 물의 삼중점 온도 (K)
    "N2": 63.15 # 질소의 삼중점 온도 (K)
}

def format_to_html(value, is_invalid=False, msg="정의되지 않음"): # 예측된 숫자를 예쁜 HTML 형식으로 바꿔주는 함수입니다.
    if is_invalid: # 물리적으로 존재하지 않는 값(초임계, 삼중점 이하 등)일 경우
        return f"<span style='color: #ff4b4b; font-size: 22px;'>{msg}</span>" # 전달받은 경고 메시지를 붉은색 텍스트로 만들어 반환합니다.
    
    s = f"{value:.4e}" # 정상적인 숫자라면 소수점 4자리까지 나오는 지수 표기법(예: 1.2345e+02)으로 문자열을 만듭니다.
    base, exp = s.split('e') # 문자열을 'e'를 기준으로 앞부분(밑)과 뒷부분(지수)으로 쪼갭니다.
    exp_int = int(exp) # 지수 부분의 글자를 실제 정수 숫자로 변환합니다.
    val_str = f"{float(base):.4f}" if exp_int == 0 else f"{float(base):.4f} × 10<sup>{exp_int}</sup>" # 지수가 0이면 그대로, 아니면 10의 거듭제곱 형태로 위첨자를 써서 꾸밉니다.
    return f"{val_str}" # 최종적으로 꾸며진 문자열을 반환합니다.

def get_html_card(name, display_val): # 화면에 물성치 결과를 네모난 카드 모양으로 그려주는 함수입니다.
    return f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 6px solid #007bff; margin-bottom: 15px; width: 100%;">
        <p style="margin: 0; font-size: 16px; color: #666; font-weight: bold;">{name}</p>
        <p style="margin: 5px 0 0 0; font-size: 28px; color: #333; font-family: 'Courier New', monospace; white-space: nowrap;">
            {display_val}
        </p>
    </div>
    """ # 카드 배경색, 모서리 둥글기, 파란색 왼쪽 테두리 등의 디자인이 적용된 HTML 코드를 반환합니다.

st.set_page_config(page_title="물리 법칙 반영 유체 예측 AI", layout="wide") # 웹 브라우저 탭의 제목을 설정하고, 화면을 좌우로 넓게 쓰도록 설정합니다.
st.title("💧 AI 기반 유체 물성 예측 모델") # 웹 페이지 최상단에 메인 제목을 크게 출력합니다.
st.write("---") # 제목 아래에 시각적으로 구분이 되도록 얇은 가로줄을 그어줍니다.

col1, col2, col3 = st.columns(3) # 화면 상단을 3개의 세로 칸(열)으로 균등하게 나눕니다.
with col1: # 첫 번째 칸에 들어갈 내용입니다.
    fluid_choice = st.selectbox("유체 종류 선택", ["Water", "N2", "Air"]) # 유체 종류를 고를 수 있는 드롭다운을 만듭니다.
with col2: # 두 번째 칸에 들어갈 내용입니다.
    unit_choice = st.selectbox("단위 선택", ["Kelvin (K)", "Celsius (°C)", "Fahrenheit (°F)"]) # 온도의 단위를 고를 수 있는 드롭다운을 만듭니다.
with col3: # 세 번째 칸에 들어갈 내용입니다.
    input_temp = st.number_input("온도 입력", value=300.0) # 사용자가 직접 온도를 타이핑해서 넣을 수 있는 입력창을 만듭니다.

# --- 온도 변환 도우미 함수들 ---
def to_k(t, u): # 입력된 온도를 AI 모델이 알아듣는 절대온도(Kelvin)로 통일시키는 함수입니다.
    if "Celsius" in u: return t + 273.15 # 섭씨면 273.15를 더해서 켈빈으로 만듭니다.
    if "Fahrenheit" in u: return (t-32)*5/9 + 273.15 # 화씨면 화씨->섭씨->켈빈 공식을 적용합니다.
    return t # 이미 켈빈이라면 그대로 돌려보냅니다.

def from_k(t_k, u): # 반대로 켈빈 온도를 사용자가 선택한 단위로 다시 돌려주는 함수입니다. (UI 출력용)
    if "Celsius" in u: return t_k - 273.15 # 섭씨가 선택되었으면 켈빈에서 273.15를 뺍니다.
    if "Fahrenheit" in u: return (t_k - 273.15) * 9/5 + 32 # 화씨가 선택되었으면 켈빈->섭씨->화씨 공식을 적용합니다.
    return t_k # 켈빈이 선택되었으면 그대로 돌려보냅니다.

def get_unit_symbol(u): # 사용자가 선택한 단위에 맞춰 문자열 기호를 반환해 주는 함수입니다.
    if "Celsius" in u: return "°C" # 섭씨일 경우 °C 반환
    if "Fahrenheit" in u: return "°F" # 화씨일 경우 °F 반환
    return "K" # 켈빈일 경우 K 반환

temp_k = to_k(input_temp, unit_choice) # 인공지능 예측과 내부 계산을 위해 입력 온도를 무조건 켈빈으로 변환해 둡니다.
unit_sym = get_unit_symbol(unit_choice) # 화면에 출력할 단위 기호(°C, °F, K)를 가져옵니다.

is_supercritical = False # 초임계 상태인지 기억할 변수 초기화
is_sub_triple = False # 삼중점 아래인지 기억할 변수 초기화

if fluid_choice in CRITICAL_TEMPS and fluid_choice in TRIPLE_TEMPS: # 물이나 질소처럼 임계/삼중점이 설정된 유체인지 검사합니다.
    tc_k = CRITICAL_TEMPS[fluid_choice] # 딕셔너리에서 켈빈 기준 임계 온도를 가져옵니다.
    tt_k = TRIPLE_TEMPS[fluid_choice] # 딕셔너리에서 켈빈 기준 삼중점 온도를 가져옵니다.
    
    # UI에 보여주기 위해 임계점과 삼중점을 사용자가 선택한 단위로 변환합니다.
    tc_disp = from_k(tc_k, unit_choice) 
    tt_disp = from_k(tt_k, unit_choice)
    
    is_supercritical = temp_k > tc_k # 입력 온도가 임계 온도보다 높은지 절대온도 기준으로 비교합니다.
    is_sub_triple = temp_k < tt_k # 입력 온도가 삼중점 온도보다 낮은지 절대온도 기준으로 비교합니다.
    
    if is_supercritical: # 만약 초임계 상태라면
        # 변환된 온도(tc_disp)와 기호(unit_sym)를 사용하여 안내창을 띄웁니다.
        st.warning(f"⚠️ 현재 온도가 {fluid_choice}의 임계 온도({tc_disp:.2f} {unit_sym})를 초과하여 초임계 상태입니다. (표면장력/증기압 소멸)")
    elif is_sub_triple: # 만약 삼중점보다 온도가 낮다면
        # 변환된 온도(tt_disp)와 기호(unit_sym)를 사용하여 안내창을 띄웁니다.
        st.warning(f"⚠️ 현재 온도가 {fluid_choice}의 삼중점({tt_disp:.2f} {unit_sym})보다 낮아 고체/기체 승화 상태입니다. (액체 표면장력 소멸)")
    else: # 평범한 액체/기체 상태라면
        # 입력한 온도(input_temp)와 변환된 삼중점/임계점을 모두 보여줍니다.
        st.info(f"💡 분석 온도: **{input_temp:.2f} {unit_sym}** (정상 액상 구간 - 삼중점: {tt_disp:.2f} {unit_sym}, 임계점: {tc_disp:.2f} {unit_sym})")
else: # 공기(Air)처럼 특별한 제약 조건이 없는 유체라면
    # 사용자가 입력한 숫자와 단위 기호만 심플하게 보여줍니다.
    st.info(f"💡 분석 온도: **{input_temp:.2f} {unit_sym}**")

# 사용자의 클릭을 유도하는 친절한 안내 문구입니다.
st.markdown("<p style='text-align: center; font-size: 16px; color: #007bff; font-weight: bold;'>👇 위 상태 안내를 확인하셨다면, 망설이지 말고 아래 버튼을 눌러 예측을 진행해 주세요! 👇</p>", unsafe_allow_html=True)

if st.button("📊 물성치 예측 실행", use_container_width=True): # 버튼을 화면 가로폭 꽉 차게 만들고 클릭 이벤트를 기다립니다.
    path = f"models/{fluid_choice.lower()}_models.pkl" # 선택한 유체의 이름으로 모델 파일 주소를 만듭니다.
    if os.path.exists(path): # 파일이 실제로 존재한다면
        models = joblib.load(path) # 인공지능 모델 묶음을 메모리로 불러옵니다.
        in_df = pd.DataFrame({'Temperature (K)': [temp_k]}) # 인공지능에게 물어볼 질문(무조건 켈빈 온도)을 표 형태로 포장합니다.
        st.success(f"### 📊 {fluid_choice} 예측 결과") # 계산 준비가 끝났다는 초록색 성공 메시지를 띄웁니다.
        
        for name, model in models.items(): # 모델 묶음에서 이름과 뇌(model)를 하나씩 꺼내옵니다.
            name_lower = name.lower() # 단어 찾기를 쉽게 하려고 이름을 소문자로 바꿉니다.
            
            if is_supercritical and ("surf" in name_lower or "tension" in name_lower or "pressure" in name_lower or "vapor" in name_lower): # 초임계 상태 예외 처리
                display_val = format_to_html(0, is_invalid=True, msg="정의되지 않음 (계면 소멸)")
            elif is_sub_triple and ("surf" in name_lower or "tension" in name_lower): # 삼중점 이하 예외 처리
                display_val = format_to_html(0, is_invalid=True, msg="측정 불가 (고체화)")
            else: # 정상적인 상태일 경우
                pred = model.predict(in_df)[0] # 인공지능에 켈빈 온도를 넣어 예측값을 뽑아냅니다.
                display_val = format_to_html(pred) # 예측된 숫자를 예쁘게 HTML로 꾸밉니다.
            
            st.markdown(get_html_card(name, display_val), unsafe_allow_html=True) # 꾸며진 결과값을 카드 형태로 화면에 출력합니다.
    else: # 모델 파일이 없다면
        st.error("모델 파일이 없습니다. q.py를 실행하세요.") # 에러 메시지를 띄웁니다.