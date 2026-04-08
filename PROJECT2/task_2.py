
import numpy as np
import pandas as pd
from task_1 import create_shape, calculate_stability 

def generate_dataset(num_samples=5000, filename='ship_data.csv'): 
    np.random.seed(42)
    data = []
    shapes = ['Rectangle', 'Triangle', 'Semicircle']
    
    print(f"[{num_samples}개 데이터 생성 시작] ")
    
    for i in range(num_samples):
        shape = np.random.choice(shapes)
       
        B = np.random.uniform(2.0, 20.0) 
        H = np.random.uniform(2.0, 20.0) if shape != 'Semicircle' else B/2
        
        poly = create_shape(shape, B, H)
        
        SG = np.random.uniform(0.05, 0.99) 
        target_area = poly.area * SG
        
        KG = np.random.uniform(0.0, 1.5 * H) 
        
        cb, bm, _ = calculate_stability(poly, target_area)
        if cb is None: 
            continue
        
        kb = cb[1]
        gm = (kb + bm) - KG 
        
        status = 1 if gm > 0 else 0
        data.append([shape, B, H, SG, KG, gm, status]) 
        
        if (i+1) % 1000 == 0:
            print(f"... {i+1}개 완료")
        
    df = pd.DataFrame(data, columns=['Shape', 'B', 'H', 'SG', 'KG', 'GM', 'Status'])
    df.to_csv(filename, index=False)
    print(f"\n데이터 생성 완료! '{filename}' 파일이 저장되었습니다.")

if __name__ == '__main__':
    generate_dataset()