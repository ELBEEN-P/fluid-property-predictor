
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import sys 

def apply_feature_engineering(df):
    X = df.copy()
    X['Draft'] = X['H'] * X['SG'] 
    X['KB_hint'] = X['Draft'] / 2.0 
    X['BM_hint'] = (X['B']**2) / (12.0 * X['Draft'] + 1e-5) 
    return X

def build_and_train_models():
    print("[1/2] AI 모델 학습을 시작합니다.")
    
    try:
        df = pd.read_csv('ship_data.csv')
    except FileNotFoundError:
        print("\n🚨 [오류] 'ship_data.csv' 파일이 없습니다. task_2.py를 먼저 실행해 주세요!\n")
        sys.exit()
        

    if len(df) < 4500:
        print("\n" + "!"*65)
        print("🚨 [치명적 문제 감지] AI가 예전 데이터(3,000개)로 학습하려 합니다!")
        print("🚨 이대로 학습하면 빙산 문제를 또 틀리게 됩니다.")
        print("🚨 터미널에서 반드시 'python task_2.py'를 먼저 실행하여 데이터를 갱신해 주세요!")
        print("!"*65 + "\n")
        sys.exit() # 엉뚱한 학습을 막기 위해 프로그램을 여기서 강제 종료시킵니다.
    
    X_base = df[['Shape', 'B', 'H', 'SG', 'KG']]
    X_engineered = apply_feature_engineering(X_base)
    
    X_encoded = pd.get_dummies(X_engineered, columns=['Shape'])
    feature_cols = X_encoded.columns
    
    y_reg = df['GM']
    y_clf = df['Status']
    
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(X_encoded, y_reg, y_clf, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    reg_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=3000, random_state=42)
    reg_model.fit(X_train_s, yr_train)
    
    yr_pred = reg_model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    nrmse = (rmse / (y_reg.max() - y_reg.min())) * 100
    
    clf_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    clf_model.fit(X_train_s, yc_train)
    
    yc_pred = clf_model.predict(X_test_s)
    accuracy = accuracy_score(yc_test, yc_pred) * 100
    
    print("\n" + "="*50)
    print("▶ 강력한 물리 특성 공학 적용 결과")
    print(f"- GM 예측 모델: R2 = {r2:.4f}, RMSE = {rmse:.4f}m (NRMSE: {nrmse:.2f}%)")
    print(f"- 전복 분류 모델: Accuracy = {accuracy:.2f}%")
    print("="*50 + "\n")
    
    return reg_model, clf_model, scaler, feature_cols

def interactive_prediction(reg_model, clf_model, scaler, feature_cols):
    print("[2/2] 실시간 선박 안정성 예측 시스템을 시작합니다.")
    
    while True:
        print("\n" + "=" * 50)
        
        while True:
            raw_shape = input("▶ 단면 형상을 입력하세요 (대소문자 구분 주의! Rectangle / Triangle / Semicircle) [종료: q]: ").strip()
            if raw_shape.lower() in ['q', 'quit', 'exit']:
                print("예측 시스템을 종료합니다. 수고하셨습니다!")
                return
            
            user_shape = raw_shape.capitalize()
            if user_shape in ['Rectangle', 'Triangle', 'Semicircle']:
                break
            else:
                print("  [경고] 정확한 형상 이름(Rectangle, Triangle, Semicircle)을 입력해주세요.\n")
                
        while True:
            try:
                user_B = float(input("▶ 선폭 B를 입력하세요 (m): "))
                if user_B > 0: break
                else: print("  [경고] 선폭 B는 0보다 커야 합니다.\n")
            except ValueError: print("  [경고] 숫자만 입력할 수 있습니다.\n")

        while True:
            try:
                user_H = float(input("▶ 선고 H를 입력하세요 (m): "))
                if user_H > 0: break
                else: print("  [경고] 선고 H는 0보다 커야 합니다.\n")
            except ValueError: print("  [경고] 숫자만 입력할 수 있습니다.\n")

        while True:
            try:
                user_SG = float(input("▶ [필수확인] 비중 SG를 입력하세요 (0.0 초과 ~ 1.0 미만): "))
                if 0.0 < user_SG < 1.0: break 
                elif user_SG >= 1.0: print("  [침몰 경고] 비중이 1.0 이상이면 배가 가라앉습니다. 다시 입력하세요!\n")
                else: print("  [경고] 비중은 0.0보다 커야 합니다.\n")
            except ValueError: print("  [경고] 숫자만 입력할 수 있습니다.\n")

        while True:
            try:
                user_KG = float(input("▶ 무게중심 높이 KG를 입력하세요 (m): "))
                if user_KG >= 0: break
                else: print("  [경고] 무게중심 높이 KG는 0 이상이어야 합니다.\n")
            except ValueError: print("  [경고] 숫자만 입력할 수 있습니다.\n")

        # 입력 데이터에도 동일하게 물리적 힌트를 적용합니다.
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
        
        status_str = "안정 (Stable)" if predicted_status == 1 else "전복 위험 (Unstable)"
        
        print("\n>>> [AI 실시간 예측 결과] <<<")
        print(f"  예측된 메타센터 높이 (GM): {predicted_gm:.4f} m")
        print(f"  예측된 선박 상태: {status_str}")

if __name__ == '__main__':
    reg, clf, scl, cols = build_and_train_models()
    interactive_prediction(reg, clf, scl, cols)