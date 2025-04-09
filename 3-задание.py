import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === 1. Загрузка и обработка реальных данных ===
real_data = pd.read_csv("C:/Users/user/Downloads/APJ064508.33-164316.0.csv")
real_data = real_data.rename(columns={'hjd': 'Time', 'mag': 'Magnitude', 'mag_err': 'Error'})
real_data = real_data.dropna()
real_data = real_data[real_data['Magnitude'] < 30]

mag0 = real_data['Magnitude'].min()
real_data['Flux'] = 10 ** (-0.4 * (real_data['Magnitude'] - mag0))

# === Обновлённый период для Sirius AB ===
P_days = 50.1 * 365.25  # = 18283.275 дней
real_data['Phase'] = (real_data['Time'] % P_days) / P_days

real_flux_min = real_data['Flux'].min()
real_flux_max = real_data['Flux'].max()

# === 2. Параметры синтетической модели Sirius AB ===
G = 39.478
m1, m2 = 2.02, 1.00
R1, R2 = 0.00795, 0.000039
L1, L2 = 25.4, 0.056
a, e, T = 19.8, 0.592, 50.1  # T в годах

# Начальные условия
r0 = a * (1 - e)
x1_0 = -r0 * m2 / (m1 + m2)
x2_0 = r0 * m1 / (m1 + m2)
v0 = np.sqrt(G * (m1 + m2) * (1 + e) / (a * (1 - e)))
vx1_0, vx2_0 = 0, 0
vy1_0 = v0 * m2 / (m1 + m2)
vy2_0 = -v0 * m1 / (m1 + m2)
y0 = [x1_0, 0, 0, x2_0, 0, 0, vx1_0, vy1_0, 0, vx2_0, vy2_0, 0]

# Уравнения движения
def equations(t, state):
    x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = state
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    ax1 = G * m2 * dx / r**3
    ay1 = G * m2 * dy / r**3
    az1 = G * m2 * dz / r**3
    ax2 = -G * m1 * dx / r**3
    ay2 = -G * m1 * dy / r**3
    az2 = -G * m1 * dz / r**3
    return [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]

# Интегрирование
t_eval = np.linspace(0, T, 5000)
sol = solve_ivp(equations, (0, T), y0, t_eval=t_eval)
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]

# Функция перекрытия
def overlap_area(R, r, d):
    if d >= R + r:
        return 0
    elif d <= abs(R - r):
        return np.pi * min(R, r)**2
    else:
        part1 = r**2 * np.arccos((d**2 + r**2 - R**2) / (2 * d * r))
        part2 = R**2 * np.arccos((d**2 + R**2 - r**2) / (2 * d * R))
        part3 = 0.5 * np.sqrt((-d + r + R)*(d + r - R)*(d - r + R)*(d + r + R))
        return part1 + part2 - part3

# Яркость модели с затмением
brightness = []
full_brightness = L1 + L2

for i in range(len(t_eval)):
    dx = x2[i] - x1[i]
    dz = z2[i] - z1[i]
    d_proj = np.sqrt(dx**2 + dz**2)

    A_overlap = overlap_area(R1, R2, d_proj)
    A1 = np.pi * R1**2
    A2 = np.pi * R2**2

    if y1[i] > y2[i]:  # B ближе
        blocked_fraction = A_overlap / A1
        total_L = L1 * (1 - blocked_fraction) + L2
    elif y2[i] > y1[i]:  # A ближе
        blocked_fraction = A_overlap / A2
        total_L = L1 + L2 * (1 - blocked_fraction)
    else:
        total_L = full_brightness

    brightness.append(total_L)

brightness = np.array(brightness)

# Нормализация по данным
model_min = brightness.min()
model_max = brightness.max()
brightness_scaled = brightness / full_brightness

# Фазы модели
synthetic_phase = (t_eval / T) % 1

# === 3. Построение графика ===
plt.figure(figsize=(10, 5))
plt.scatter(real_data['Phase'], real_data['Flux'], s=10, alpha=0.6, label='Наблюд. данные (фаза)', color='royalblue')
plt.plot(synthetic_phase, brightness_scaled, label='Синтетическая модель (Sirius AB)', color='darkorange')
plt.xlabel("Фаза орбиты")
plt.ylabel("Яркость (отн. ед.)")
plt.title("Фазовая кривая блеска Sirius AB: Наблюдения и модель")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
