import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры
G = 39.478
m1, m2 = 2.02, 1.00
R1, R2 = 0.00795, 0.000039  # Радиусы в а.е.
L1, L2 = 25.4, 0.056        # Светимости
a, e, T = 19.8, 0.592, 50.1

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

# Интегрируем
t_eval = np.linspace(0, T, 5000)
sol = solve_ivp(equations, (0, T), y0, t_eval=t_eval)
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]

# Функция перекрытия кругов
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

# Вычисляем яркость
brightness = []
full_brightness = L1 + L2  # Полная яркость, когда оба видны

for i in range(len(t_eval)):
    dx = x2[i] - x1[i]
    dz = z2[i] - z1[i]
    d_proj = np.sqrt(dx**2 + dz**2)

    A_overlap = overlap_area(R1, R2, d_proj)
    A1 = np.pi * R1**2
    A2 = np.pi * R2**2

    if y1[i] > y2[i]:  # B ближе, может закрыть A
        blocked_fraction = A_overlap / A1
        total_L = L1 * (1 - blocked_fraction) + L2
    elif y2[i] > y1[i]:  # A ближе, может закрыть B
        blocked_fraction = A_overlap / A2
        total_L = L1 + L2 * (1 - blocked_fraction)
    else:
        total_L = full_brightness

    brightness.append(total_L)

brightness = np.array(brightness)

# Строим график
plt.figure(figsize=(10, 5))
plt.plot(t_eval, brightness, label='Суммарная яркость (с затмениями)', color='black')
plt.axhline(full_brightness, color='gray', linestyle='--', label='Полная яркость (без затмений)')
plt.xlabel('Время (годы)')
plt.ylabel('Яркость (отн. ед.)')
plt.title('Синтетическая кривая блеска Сириуса AB')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

