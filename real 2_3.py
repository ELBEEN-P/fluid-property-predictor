import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

# 폴더 생성 (결과물 저장용)
os.makedirs('models', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

# 파일명 매핑 (웹 UI와의 호환성을 위해 key는 짧게 유지)
fluid_files = {
    'water': 'water.csv',
    'n2': 'n2.csv',
    'air': 'air_properties_data.csv'
}

for fluid_name, file_name in fluid_files.items():
    print(f"\n{'='*40}")
    print(f"--- {fluid_name.upper()} 데이터 학습 및 평가 시작 ---")
    
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"⚠️ 에러: {file_name} 파일이 없습니다. 폴더 위치를 확인해주세요.")
        continue
        
    # 쓸데없는 빈 열(Unnamed) 제거
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    temp_col = 'Temperature (K)'
    if temp_col not in df.columns:
        print(f"에러: {temp_col} 컬럼을 찾을 수 없습니다. 헤더를 확인해주세요.")
        continue

    target_columns = [col for col in df.columns if col != temp_col]
    fluid_models = {}
    
    # 물성치별 개별 모델 학습 (데이터 손실 방지 핵심 로직)
    for target in target_columns:
        print(f"\n▶ [{target}] 모델 학습 중...")
        
        # 해당 물성치에 결측치가 없는 유효 데이터만 추출
        subset = df[[temp_col, target]].dropna()
        
        if len(subset) < 10:
            print(f"  -> 유효 데이터 부족({len(subset)}개). 학습 건너뜀.")
            continue
            
        print(f"  -> 유효 데이터: {len(subset)}개")
        
        X = subset[[temp_col]]
        y = subset[target]
        
        # 8:2 분할 및 Random Forest 학습
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 평가 지표 계산
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  -> R^2 점수: {r2:.4f}, RMSE: {rmse:.4e}")
        
        fluid_models[target] = model
        
        # 오차 분석 그래프 생성 (A등급 보고서용)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')
        
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        
        plt.title(f'{fluid_name.upper()} - {target}\nR^2: {r2:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        
        safe_col_name = target.split(' ')[0].replace('/', '_')
        plt.tight_layout()
        plt.savefig(f'graphs/{fluid_name}_{safe_col_name}_error_plot.png')
        plt.close()
        
    # 모델 저장 (.pkl)
    if fluid_models:
        joblib.dump(fluid_models, f'models/{fluid_name}_models.pkl')
        print(f"\n✅ {fluid_name.upper()} 모델 및 그래프 저장 완료")