import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from shapely.affinity import rotate
from task_1 import create_shape, calculate_stability

def export_animation(shape='Rectangle', B=8, H=6, SG=0.5, KG=1.5):
    print("애니메이션 렌더링을 시작합니다. 잠시만 기다려주세요...")
    poly = create_shape(shape, B, H)
    target_area = poly.area * SG
    
    thetas = np.linspace(-45, 45, 90)
    gz_values = []
    
    for th in thetas:
        rotated_poly = rotate(poly, th, origin=(0,0))
        CB, _, _ = calculate_stability(rotated_poly, target_area)
        if CB is None:
            gz_values.append(0)
            continue
            
        CG_x = -KG * np.sin(np.radians(th))
        GZ = CB[0] - CG_x
        gz_values.append(GZ)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-50, 50)
    ax.set_ylim(min(gz_values)*1.2, max(gz_values)*1.2)
    ax.set_xlabel("Heel Angle (degrees)")
    ax.set_ylabel("Righting Arm GZ (m)")
    ax.set_title(f"Dynamic GZ Curve ({shape})")
    ax.grid(True)
    ax.axhline(0, color='black', linewidth=1)
    
    line, = ax.plot([], [], 'b-', lw=2)
    point, = ax.plot([], [], 'ro', markersize=8)
    
    def animate(i):
        line.set_data(thetas[:i+1], gz_values[:i+1])
        point.set_data([thetas[i]], [gz_values[i]])
        return line, point
        
    ani = animation.FuncAnimation(fig, animate, frames=len(thetas), interval=50, blit=True)
    ani.save('stability_animation.gif', writer='pillow')
    print("성공적으로 'stability_animation.gif' 파일이 생성되었습니다.")

if __name__ == '__main__':
    export_animation()