import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Параметры задачи
G = 39.478  # Гравитационная постоянная в а.е.^3 / (M☉ * год^2)
m1 = 2.02  # Масса Сириуса A в массах Солнца
m2 = 1.00  # Масса Сириуса B в массах Солнца
a = 19.8  # Большая полуось орбиты в а.е.
e = 0.592  # Эксцентриситет орбиты
T = 50.1  # Орбитальный период в годах

# Начальные условия (перицентр)
r0 = a * (1 - e)
x1_0, y1_0, z1_0 = -r0 * m2 / (m1 + m2), 0, 0
x2_0, y2_0, z2_0 = r0 * m1 / (m1 + m2), 0, 0
v0 = np.sqrt(G * (m1 + m2) * (1 + e) / (a * (1 - e)))
vx1_0, vy1_0, vz1_0 = 0, v0 * m2 / (m1 + m2), 0
vx2_0, vy2_0, vz2_0 = 0, -v0 * m1 / (m1 + m2), 0

# Функция для вычисления ускорений
def equations(t, state):
    x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2 = state
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    ax1 = G * m2 * (x2 - x1) / r12**3
    ay1 = G * m2 * (y2 - y1) / r12**3
    az1 = G * m2 * (z2 - z1) / r12**3
    ax2 = -G * m1 * (x2 - x1) / r12**3
    ay2 = -G * m1 * (y2 - y1) / r12**3
    az2 = -G * m1 * (z2 - z1) / r12**3
    return [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]

# Интегрирование
t_span = (0, T)
y0 = [x1_0, y1_0, z1_0, x2_0, y2_0, z2_0, vx1_0, vy1_0, vz1_0, vx2_0, vy2_0, vz2_0]
t_eval = np.linspace(0, T, 5000)
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval)

# Координаты звезд
x1, y1, z1 = sol.y[0], sol.y[1], sol.y[2]
x2, y2, z2 = sol.y[3], sol.y[4], sol.y[5]

# Графики орбит
plt.figure(figsize=(12, 5))

# Вид сверху
plt.subplot(1, 2, 1)
plt.plot(x1, y1, label='Сириус A', color='b')
plt.plot(x2, y2, label='Сириус B', color='r')
plt.scatter(0, 0, color='black', marker='x', label='Центр масс')
plt.xlabel('X (а.е.)')
plt.ylabel('Y (а.е.)')
plt.title('Орбита (вид сверху)')
plt.legend()

# Вид сбоку
plt.subplot(1, 2, 2)
plt.plot(x1, z1, label='Сириус A', color='b')
plt.plot(x2, z2, label='Сириус B', color='r')
plt.scatter(0, 0, color='black', marker='x', label='Центр масс')
plt.xlabel('X (а.е.)')
plt.ylabel('Z (а.е.)')
plt.title('Орбита (вид сбоку)')
plt.legend()

plt.tight_layout()
plt.show()

