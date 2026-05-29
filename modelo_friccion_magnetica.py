import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Modelo reducido de fricción magnética sin contacto
# Dos rotores / dos subredes: theta_A y theta_B
# ============================================================

# Parámetros generales
gamma = 1.0          # coeficiente de fricción rotacional
v = 0.05             # velocidad de deslizamiento
dt = 0.01            # paso temporal
n_periods = 6        # número de periodos de red simulados
steps_per_period = int(1 / (v * dt))
n_steps = n_periods * steps_per_period

# Desplazamiento total
t = np.arange(n_steps) * dt
x = v * t            # x está medido en unidades del periodo de red

# Barrido de separaciones
h_values = np.linspace(0.5, 5.0, 40)


def wrap_angle(theta):
    """
    Lleva un ángulo al intervalo [-pi, pi].
    Esto ayuda a evitar saltos artificiales en las gráficas.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def phi_substrate(x):
    """
    Dirección efectiva del campo magnético del sustrato.
    Se toma periódica en el desplazamiento.
    """
    return 2 * np.pi * x


def B_of_h(h):
    """
    Intensidad de la interacción con el sustrato.
    Disminuye con la separación h.
    """
    return np.exp(-h)


def J_of_h(h):
    """
    Intensidad de la interacción interna entre rotores.

    En este modelo simple, la interacción entre rotores se toma
    relativamente más importante cuando aumenta h.
    """
    return 0.35 * (1 - np.exp(-h))


def simulate(h, theta_A0=0.2, theta_B0=-0.2):
    """
    Simula la dinámica de dos rotores magnéticos para una separación h.

    Energía efectiva:

    U = J(h) cos(theta_A - theta_B)
        - B(h)[cos(theta_A - phi) + cos(theta_B - phi)]

    Nota:
    - El término J cos(theta_A - theta_B) favorece configuración antiparalela
      cuando J > 0, porque su mínimo ocurre cerca de theta_A - theta_B = pi.
    - El término con B favorece alineación de ambos rotores con el campo
      periódico del sustrato.
    """

    B = B_of_h(h)
    J = J_of_h(h)

    theta_A = np.zeros(n_steps)
    theta_B = np.zeros(n_steps)

    theta_A[0] = theta_A0
    theta_B[0] = theta_B0

    dissipation_rate = np.zeros(n_steps)

    for k in range(n_steps - 1):
        phi = phi_substrate(x[k])

        # Derivadas de U respecto a los ángulos
        dU_dtheta_A = -J * np.sin(theta_A[k] - theta_B[k]) + B * np.sin(theta_A[k] - phi)
        dU_dtheta_B =  J * np.sin(theta_A[k] - theta_B[k]) + B * np.sin(theta_B[k] - phi)

        # Dinámica sobreamortiguada:
        # gamma * theta_dot = - dU/dtheta
        theta_A_dot = -dU_dtheta_A / gamma
        theta_B_dot = -dU_dtheta_B / gamma

        theta_A[k + 1] = theta_A[k] + theta_A_dot * dt
        theta_B[k + 1] = theta_B[k] + theta_B_dot * dt

        # Potencia disipada por fricción rotacional
        dissipation_rate[k] = gamma * (theta_A_dot**2 + theta_B_dot**2)

    # Último valor aproximado
    dissipation_rate[-1] = dissipation_rate[-2]

    # Parámetro de orden:
    # Sigma = 1  -> orden ferromagnético, rotores paralelos
    # Sigma = -1 -> orden antiferromagnético, rotores antiparalelos
    Sigma = np.cos(theta_A - theta_B)

    # Energía disipada total
    E_diss = np.trapz(dissipation_rate, t)

    # Fricción efectiva: energía disipada por distancia recorrida
    total_distance = x[-1] - x[0]
    F_mag = E_diss / total_distance

    return {
        "h": h,
        "x": x,
        "theta_A": theta_A,
        "theta_B": theta_B,
        "Sigma": Sigma,
        "dissipation_rate": dissipation_rate,
        "E_diss": E_diss,
        "F_mag": F_mag,
        "B": B,
        "J": J
    }


# ============================================================
# Simulaciones para tres regímenes
# ============================================================

h_small = 0.7       # domina el sustrato
h_middle = 2.0      # régimen competitivo
h_large = 4.5       # domina interacción interna

result_small = simulate(h_small)
result_middle = simulate(h_middle)
result_large = simulate(h_large)


# ============================================================
# Figura 1: Ángulos vs desplazamiento
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(result_middle["x"], wrap_angle(result_middle["theta_A"]), label=r"$\theta_A$")
plt.plot(result_middle["x"], wrap_angle(result_middle["theta_B"]), label=r"$\theta_B$")

plt.xlabel(r"Desplazamiento $x$ [periodos de red]")
plt.ylabel(r"Ángulo del rotor [rad]")
plt.title(r"Dinámica angular de dos rotores en el régimen competitivo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_1_angulos_vs_desplazamiento.png", dpi=300)
plt.show()


# ============================================================
# Figura 2: Parámetro de orden para tres separaciones
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(result_small["x"], result_small["Sigma"], label=f"h = {h_small:.1f}  sustrato dominante")
plt.plot(result_middle["x"], result_middle["Sigma"], label=f"h = {h_middle:.1f}  régimen competitivo")
plt.plot(result_large["x"], result_large["Sigma"], label=f"h = {h_large:.1f}  interacción interna dominante")

plt.xlabel(r"Desplazamiento $x$ [periodos de red]")
plt.ylabel(r"Parámetro de orden $\Sigma = \cos(\theta_A-\theta_B)$")
plt.title(r"Transición entre orden ferromagnético y antiferromagnético")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_2_parametro_de_orden.png", dpi=300)
plt.show()


# ============================================================
# Figura 3: Fricción magnética efectiva vs separación h
# ============================================================

F_values = []
Sigma_mean_values = []
B_values = []
J_values = []

for h in h_values:
    result = simulate(h)
    F_values.append(result["F_mag"])
    Sigma_mean_values.append(np.mean(result["Sigma"]))
    B_values.append(result["B"])
    J_values.append(result["J"])

F_values = np.array(F_values)
Sigma_mean_values = np.array(Sigma_mean_values)
B_values = np.array(B_values)
J_values = np.array(J_values)

plt.figure(figsize=(9, 5))

plt.plot(h_values, F_values, marker="o")

plt.xlabel(r"Separación entre capas $h$ [unidades arbitrarias]")
plt.ylabel(r"Fricción magnética efectiva $F_{\mathrm{mag}}$")
plt.title(r"Fricción magnética no monótona en el modelo reducido")
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_3_friccion_vs_h.png", dpi=300)
plt.show()


# ============================================================
# Figura 4 opcional: Comparación entre interacciones B(h) y J(h)
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(h_values, B_values, label=r"$B(h)$ interacción con el sustrato")
plt.plot(h_values, J_values, label=r"$J(h)$ interacción interna")

plt.xlabel(r"Separación $h$")
plt.ylabel("Intensidad de interacción")
plt.title("Competencia entre interacción con el sustrato e interacción interna")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_4_competencia_interacciones.png", dpi=300)
plt.show()


# ============================================================
# Resumen numérico
# ============================================================

h_peak = h_values[np.argmax(F_values)]
F_peak = np.max(F_values)

print("=========================================")
print("Resumen del modelo reducido")
print("=========================================")
print(f"Separación con máxima fricción: h = {h_peak:.3f}")
print(f"Fricción máxima efectiva: F_mag = {F_peak:.5f}")
print()
print("Archivos generados:")
print("- figura_1_angulos_vs_desplazamiento.png")
print("- figura_2_parametro_de_orden.png")
print("- figura_3_friccion_vs_h.png")
print("- figura_4_competencia_interacciones.png")