import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Modelo reducido de fricción magnética sin contacto
# Dos rotores / dos subredes: theta_A y theta_B
# ============================================================
#
# Objetivo del código:
# Construir un modelo mínimo que reproduzca cualitativamente la idea central
# del artículo: la fricción magnética puede ser máxima cuando compiten
# la interacción con el sustrato y la interacción interna entre rotores.
#
# Importante:
# Este código NO pretende reproducir cuantitativamente el experimento completo
# de 7x7 rotores. Es un modelo reducido útil para el paper del curso.
# ============================================================


# ============================================================
# Parámetros generales de la simulación
# ============================================================

gamma = 1.0          # coeficiente de fricción rotacional
v = 0.05             # velocidad de deslizamiento en periodos de red / unidad de tiempo
dt = 0.01            # paso temporal
n_periods = 8        # número de periodos de red simulados
steps_per_period = int(1 / (v * dt))
n_steps = n_periods * steps_per_period

# Tiempo y desplazamiento. La variable x está medida en periodos de red.
t = np.arange(n_steps) * dt
x = v * t

# Valores de separación h que se barren para obtener F_mag(h)
h_values = np.linspace(0.5, 5.0, 70)

# Pequeño número para evitar divisiones por cero
eps = 1e-12


# ============================================================
# Funciones auxiliares
# ============================================================

def wrap_angle(theta):
    """
    Lleva un ángulo al intervalo [-pi, pi].
    Esto ayuda a visualizar las orientaciones sin saltos de 2pi.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def phi_substrate(x_value):
    """
    Dirección efectiva del campo magnético del sustrato.
    El campo se toma periódico con el desplazamiento.
    """
    return 2 * np.pi * x_value


def B_of_h(h):
    """
    Intensidad de la interacción con el sustrato.
    Disminuye al aumentar la separación entre capas.
    """
    return np.exp(-h)


def J_of_h(h):
    """
    Intensidad efectiva de la interacción interna entre rotores.
    En este modelo reducido se hace crecer relativamente con h, para representar
    que a grandes separaciones domina la interacción interna frente al sustrato.
    """
    return 0.35 * (1 - np.exp(-h))


def competition_factor(B, J):
    """
    Factor fenomenológico de competencia entre interacciones.

    C = 1 cuando B = J.
    C tiende a 0 cuando una interacción domina completamente sobre la otra.

    Este factor permite resaltar que la disipación magnética relevante aparece
    principalmente cuando el sustrato y la interacción interna son comparables.
    """
    return 4 * B * J / ((B + J) ** 2 + eps)


# ============================================================
# Simulación dinámica
# ============================================================

def simulate(h, theta_A0=0.0, theta_B0=0.05):
    """
    Simula la dinámica sobreamortiguada de dos rotores magnéticos.

    Variables dinámicas:
    theta_A(t), theta_B(t)

    Energía efectiva:

    U(theta_A, theta_B, x, h) =
        J(h) cos(theta_A - theta_B)
        - B(h)[cos(theta_A - phi(x)) + cos(theta_B - phi(x))]

    Interpretación:
    - El término con B(h) favorece que ambos rotores sigan el campo periódico
      del sustrato.
    - El término con J(h) favorece una configuración antiparalela entre rotores.
    - Cuando B y J son comparables, hay competencia entre ambas tendencias.

    Ecuaciones de movimiento sobreamortiguadas:

        gamma * theta_dot = - dU/dtheta

    La potencia disipada por fricción rotacional se calcula como:

        P_diss = gamma * (theta_A_dot^2 + theta_B_dot^2)
    """

    B = B_of_h(h)
    J = J_of_h(h)

    theta_A = np.zeros(n_steps)
    theta_B = np.zeros(n_steps)
    dissipation_rate = np.zeros(n_steps)

    theta_A[0] = theta_A0
    theta_B[0] = theta_B0

    for k in range(n_steps - 1):
        phi = phi_substrate(x[k])

        # Derivadas parciales de U respecto a theta_A y theta_B
        dU_dtheta_A = -J * np.sin(theta_A[k] - theta_B[k]) + B * np.sin(theta_A[k] - phi)
        dU_dtheta_B =  J * np.sin(theta_A[k] - theta_B[k]) + B * np.sin(theta_B[k] - phi)

        # Dinámica sobreamortiguada
        theta_A_dot = -dU_dtheta_A / gamma
        theta_B_dot = -dU_dtheta_B / gamma

        theta_A[k + 1] = theta_A[k] + theta_A_dot * dt
        theta_B[k + 1] = theta_B[k] + theta_B_dot * dt

        # Potencia disipada por fricción rotacional interna
        dissipation_rate[k] = gamma * (theta_A_dot**2 + theta_B_dot**2)

    dissipation_rate[-1] = dissipation_rate[-2]

    # Parámetro de orden:
    # Sigma = +1  -> rotores paralelos, orden tipo ferromagnético.
    # Sigma = -1  -> rotores antiparalelos, orden tipo antiferromagnético.
    Sigma = np.cos(theta_A - theta_B)

    # Energía disipada total y fricción rotacional promedio
    E_diss = np.trapz(dissipation_rate, t)
    total_distance = x[-1] - x[0]
    F_rot = E_diss / total_distance

    # Factor de competencia y fricción magnética efectiva corregida.
    # Esta es la magnitud que se usa para representar el mecanismo cualitativo
    # de fricción máxima en el régimen competitivo.
    C = competition_factor(B, J)
    F_mag = F_rot * C

    return {
        "h": h,
        "x": x,
        "theta_A": theta_A,
        "theta_B": theta_B,
        "Sigma": Sigma,
        "dissipation_rate": dissipation_rate,
        "E_diss": E_diss,
        "F_rot": F_rot,
        "F_mag": F_mag,
        "B": B,
        "J": J,
        "C": C,
    }


# ============================================================
# Simulaciones representativas de tres regímenes
# ============================================================

# Estos valores se escogieron para representar tres zonas:
# 1. Sustrato dominante: B >> J
# 2. Régimen competitivo: B aproximadamente comparable con J
# 3. Interacción interna dominante: J >> B
h_small = 0.7
h_middle = 1.35
h_large = 4.5

result_small = simulate(h_small)
result_middle = simulate(h_middle)
result_large = simulate(h_large)


# ============================================================
# Figura 1: Ángulos vs desplazamiento en el régimen competitivo
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
plt.plot(result_small["x"], result_small["Sigma"], label=f"h = {h_small:.2f}  sustrato dominante")
plt.plot(result_middle["x"], result_middle["Sigma"], label=f"h = {h_middle:.2f}  régimen competitivo")
plt.plot(result_large["x"], result_large["Sigma"], label=f"h = {h_large:.2f}  interacción interna dominante")
plt.xlabel(r"Desplazamiento $x$ [periodos de red]")
plt.ylabel(r"Parámetro de orden $\Sigma = \cos(\theta_A-\theta_B)$")
plt.title(r"Orden magnético efectivo en tres regímenes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_2_parametro_de_orden.png", dpi=300)
plt.show()


# ============================================================
# Barrido en h para calcular F_mag(h)
# ============================================================

F_rot_values = []
F_mag_values = []
Sigma_mean_values = []
B_values = []
J_values = []
C_values = []

for h in h_values:
    result = simulate(h)
    F_rot_values.append(result["F_rot"])
    F_mag_values.append(result["F_mag"])
    Sigma_mean_values.append(np.mean(result["Sigma"]))
    B_values.append(result["B"])
    J_values.append(result["J"])
    C_values.append(result["C"])

F_rot_values = np.array(F_rot_values)
F_mag_values = np.array(F_mag_values)
Sigma_mean_values = np.array(Sigma_mean_values)
B_values = np.array(B_values)
J_values = np.array(J_values)
C_values = np.array(C_values)


# ============================================================
# Figura 3: Fricción magnética efectiva vs separación h
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(h_values, F_mag_values, marker="o", label=r"$F_{\mathrm{mag}} = F_{\mathrm{rot}} C(h)$")
plt.xlabel(r"Separación entre capas $h$ [unidades arbitrarias]")
plt.ylabel(r"Fricción magnética efectiva $F_{\mathrm{mag}}$")
plt.title(r"Fricción magnética con máximo en el régimen competitivo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_3_friccion_vs_h.png", dpi=300)
plt.show()


# ============================================================
# Figura 4: Competencia entre interacciones B(h), J(h) y C(h)
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(h_values, B_values, label=r"$B(h)$ interacción con el sustrato")
plt.plot(h_values, J_values, label=r"$J(h)$ interacción interna")
plt.plot(h_values, C_values, linestyle="--", label=r"$C(h)$ factor de competencia")
plt.xlabel(r"Separación $h$")
plt.ylabel("Intensidad relativa")
plt.title("Competencia entre interacción con el sustrato e interacción interna")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_4_competencia_interacciones.png", dpi=300)
plt.show()


# ============================================================
# Figura 5: Comparación entre disipación rotacional cruda y fricción efectiva
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(h_values, F_rot_values, marker="o", label=r"$F_{\mathrm{rot}}$ disipación rotacional cruda")
plt.plot(h_values, F_mag_values, marker="s", label=r"$F_{\mathrm{mag}}$ fricción efectiva con competencia")
plt.xlabel(r"Separación $h$")
plt.ylabel("Magnitud efectiva")
plt.title("Efecto del factor de competencia sobre la fricción magnética")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figura_5_comparacion_fricciones.png", dpi=300)
plt.show()


# ============================================================
# Resumen numérico
# ============================================================

h_peak = h_values[np.argmax(F_mag_values)]
F_peak = np.max(F_mag_values)
B_peak = B_of_h(h_peak)
J_peak = J_of_h(h_peak)
C_peak = competition_factor(B_peak, J_peak)

print("=========================================")
print("Resumen del modelo reducido")
print("=========================================")
print(f"Separación con máxima fricción efectiva: h = {h_peak:.3f}")
print(f"Fricción magnética efectiva máxima: F_mag = {F_peak:.5f}")
print(f"B(h_peak) = {B_peak:.5f}")
print(f"J(h_peak) = {J_peak:.5f}")
print(f"C(h_peak) = {C_peak:.5f}")
print()
print("Archivos generados:")
print("- figura_1_angulos_vs_desplazamiento.png")
print("- figura_2_parametro_de_orden.png")
print("- figura_3_friccion_vs_h.png")
print("- figura_4_competencia_interacciones.png")
print("- figura_5_comparacion_fricciones.png")