import numpy as np
import matplotlib.pyplot as plt

# ФИЗИЧЕСКИЕ КОНСТАНТЫ
g = 9.81
R = 6.37e6
rho0 = 1.225
beta = 0.125 / 1000

# ПАРАМЕТРЫ РАКЕТЫ
m0 = 6 * 10**5   # начальная масса
m_cons = 1 * 10**4   # конечная масса
alpha = 1000   # расход топлива
u = 2500    # скорость истечения

c = 0.5  # коэффициент сопротивления
S = 10  # площадь сечения

# ЧИСЛЕННЫЕ ПАРАМЕТРЫ
dt = 0.05  # шаг по времени
t_max = 600  # время моделировани
N = int(t_max / dt)

# НАЧАЛЬНЫЕ УСЛОВИЯ
x = 0.0
y = 0.0
v = 1.0  # малая начальная скорость
theta = np.pi / 2  # вертикальный старт

# МАССИВЫ ДЛЯ ХРАНЕНИЯ РЕЗУЛЬТАТОВ
T = []
X = []
Y = []
V = []
M = []
A = []

# ОСНОВНОЙ ЦИКЛ МОДЕЛИРОВАНИЯ
for i in range(N):
    t = i * dt

    # масса
    m = max(m0 - alpha * t, m_cons)

    # плотность воздуха
    rho = rho0 * 10 ** (-beta * y)

    # сопротивление
    k = 0.5 * c * S * rho
    F_drag = k * v**2

    # гравитация
    g_eff = g * (R / (R + y))**2

    # производные
    dv = (alpha * u - F_drag) / m - g_eff * np.sin(theta)
    dtheta = -g_eff * np.cos(theta) / max(v, 1e-3)

    dx = v * np.cos(theta)
    dy = v * np.sin(theta)

    # шаг Эйлера
    v += dv * dt
    theta += dtheta * dt
    x += dx * dt
    y += dy * dt

    # ускорение
    a = dv

    # сохранение
    T.append(t)
    X.append(x)
    Y.append(y)
    V.append(v)
    M.append(m)
    A.append(a)

    # остановка при падении
    if y < 0:
        break

# ПОСТРОЕНИЕ ГРАФИКОВ
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.plot(T, A)
plt.title("a(t)")
plt.xlabel("t, с")
plt.ylabel("a, м/с²")

plt.subplot(2, 3, 2)
plt.plot(T, V)
plt.title("v(t)")
plt.xlabel("t, с")
plt.ylabel("v, м/с")

plt.subplot(2, 3, 3)
plt.plot(T, M)
plt.title("m(t)")
plt.xlabel("t, с")
plt.ylabel("m, кг")

plt.subplot(2, 3, 4)
plt.plot(X, Y)
plt.title("Траектория y(x)")
plt.xlabel("x, м")
plt.ylabel("y, м")

plt.subplot(2, 3, 5)
plt.plot(Y, V)
plt.title("v(y)")
plt.xlabel("y, м")
plt.ylabel("v, м/с")

plt.tight_layout()
plt.show()

# ПРОВЕРКА ПЕРВОЙ КОСМИЧЕСКОЙ
v_orb = np.sqrt(g * R)
print(f"Максимальная скорость: {max(V):.1f} м/с")
print(f"Первая космическая:     {v_orb:.1f} м/с")
