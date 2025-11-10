import matplotlib.pyplot as plt
import numpy as np

# Valores de X (mesmos pontos do script original)
x = np.array(
    [
        0,
        0.055,
        0.105,
        0.155,
        0.205,
        0.255,
        0.305,
        0.355,
        0.405,
        0.455,
        0.505,
        0.555,
        0.605,
        0.655,
        0.705,
        0.755,
        0.805,
        0.855,
        0.905,
        0.955,
        0.985,
        1,
    ]
)

# Dados de erro relativo wa
erro_wa_05_BDF = np.array(
    [
        0.00e00,
        5.09e-06,
        8.23e-06,
        7.20e-06,
        9.03e-06,
        7.76e-06,
        9.63e-06,
        1.17e-05,
        1.05e-05,
        1.46e-05,
        1.49e-05,
        1.76e-05,
        1.93e-05,
        1.89e-05,
        2.52e-05,
        3.33e-05,
        4.04e-05,
        5.67e-05,
        8.71e-05,
        1.88e-04,
        5.75e-04,
        0.00e00,
    ]
)

erro_wa_075_BDF = np.array(
    [
        0.00e00,
        9.51e-06,
        9.78e-06,
        1.17e-05,
        1.28e-05,
        1.31e-05,
        1.49e-05,
        1.69e-05,
        1.81e-05,
        2.19e-05,
        2.46e-05,
        2.80e-05,
        3.30e-05,
        3.99e-05,
        4.79e-05,
        5.87e-05,
        7.77e-05,
        1.04e-04,
        1.64e-04,
        3.68e-04,
        1.13e-03,
        0.00e00,
    ]
)

erro_wa_095_BDF = np.array(
    [
        0.00e00,
        7.63e-06,
        8.30e-06,
        9.70e-06,
        1.11e-05,
        1.36e-05,
        1.67e-05,
        1.94e-05,
        2.32e-05,
        2.69e-05,
        3.32e-05,
        4.02e-05,
        4.90e-05,
        6.09e-05,
        7.82e-05,
        1.02e-04,
        1.39e-04,
        2.03e-04,
        3.34e-04,
        7.55e-04,
        2.37e-03,
        0.00e00,
    ]
)

erro_wa_05_RSTF = np.array(
    [
        0.00e00,
        5.09e-06,
        6.07e-06,
        7.20e-06,
        9.03e-06,
        7.76e-06,
        9.63e-06,
        1.17e-05,
        1.05e-05,
        1.14e-05,
        1.49e-05,
        1.76e-05,
        1.93e-05,
        1.89e-05,
        2.52e-05,
        3.33e-05,
        4.04e-05,
        5.57e-05,
        8.71e-05,
        1.88e-04,
        5.75e-04,
        0.00e00,
    ]
)

erro_wa_075_RSTF = np.array(
    [
        0.00e00,
        8.14e-06,
        9.78e-06,
        1.03e-05,
        1.13e-05,
        1.16e-05,
        1.16e-05,
        1.35e-05,
        1.63e-05,
        1.82e-05,
        2.06e-05,
        2.36e-05,
        2.82e-05,
        3.46e-05,
        4.20e-05,
        5.17e-05,
        7.34e-05,
        9.85e-05,
        1.64e-04,
        3.65e-04,
        1.13e-03,
        0.00e00,
    ]
)

erro_wa_095_RSTF = np.array(
    [
        0.00e00,
        5.51e-06,
        5.08e-06,
        5.35e-06,
        4.46e-06,
        4.65e-06,
        5.29e-06,
        6.58e-06,
        7.59e-06,
        9.52e-06,
        1.25e-05,
        1.57e-05,
        2.16e-05,
        3.14e-05,
        4.41e-05,
        6.52e-05,
        9.79e-05,
        1.58e-04,
        2.90e-04,
        7.16e-04,
        2.38e-03,
        0.00e00,
    ]
)

# Plot
plt.figure(figsize=(12, 8))

# Dados de erro - com cores na mesma vibe
plt.plot(
    x,
    erro_wa_05_BDF,
    "o-",
    color="coral",
    markersize=5,
    label="Erro wa (wa0=0.5) - BDF",
    alpha=0.8,
)
plt.plot(
    x,
    erro_wa_075_BDF,
    "o-",
    color="tomato",
    markersize=5,
    label="Erro wa (wa0=0.75) - BDF",
    alpha=0.8,
)
plt.plot(
    x,
    erro_wa_095_BDF,
    "o-",
    color="firebrick",
    markersize=5,
    label="Erro wa (wa0=0.95) - BDF",
    alpha=0.8,
)
plt.plot(
    x,
    erro_wa_05_RSTF,
    "o-",
    color="deepskyblue",
    markersize=5,
    label="Erro wa (wa0=0.5) - RSTF",
    alpha=0.8,
)
plt.plot(
    x,
    erro_wa_075_RSTF,
    "o-",
    color="royalblue",
    markersize=5,
    label="Erro wa (wa0=0.75) - RSTF",
    alpha=0.8,
)
plt.plot(
    x,
    erro_wa_095_RSTF,
    "o-",
    color="darkblue",
    markersize=5,
    label="Erro wa (wa0=0.95) - RSTF",
    alpha=0.8,
)

plt.xlabel("Posição (z)")
plt.ylabel("Erro Relativo wa")
plt.title("Erro Relativo wa para Diferentes Condições de Contorno e Solvers")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 0.003)  # Ajustado para melhor visualização dos erros
plt.yscale("linear")  # Você pode mudar para 'log' se preferir escala logarítmica
plt.tight_layout()
plt.show()
