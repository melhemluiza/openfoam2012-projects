import numpy as np
import matplotlib.pyplot as plt

# Dados da PRIMEIRA imagem (rho_a numérico) - CORRETO!
# Valores de X (cada um ocupa duas linhas mas é um número único)
x_num_a = np.array([0, 0.055, 0.105, 0.155, 0.205, 0.255, 0.305, 0.355, 0.405, 0.455,
    0.505, 0.555, 0.605, 0.655, 0.705, 0.755, 0.805, 0.855, 0.905, 0.955,
    0.985, 1])

# Valores de Y (linhas individuais após cada X)
y_num_a = np.array([0.9, 0.88649, 0.87264, 0.8571, 0.83966, 0.8201, 0.79815, 0.77352,
    0.74589, 0.71488, 0.68009, 0.64105, 0.59726, 0.54811, 0.49298,
    0.43111, 0.36169, 0.28381, 0.19642, 0.09837, 0.033887, 0])

# Dados analíticos (já estavam corretos)
x_ana_a = np.array([0, 0.055, 0.105, 0.155, 0.205, 0.255, 0.305, 0.355, 0.405, 0.455,
    0.505, 0.555, 0.605, 0.655, 0.705, 0.755, 0.805, 0.855, 0.905, 0.955,
    0.985, 1])

y_ana_a = np.array([ 0.9000014907, 0.8865005173, 0.872651391, 0.857112404, 0.8396773609,
   0.8201149063, 0.7981654547, 0.7735377467, 0.7459049832, 0.7149004895,
   0.6801128495, 0.6410804463, 0.597285337, 0.5481463795, 0.4930115212,
   0.4311491467, 0.3617383692, 0.2838581377, 0.1964750157, 0.09842946718,
   0.03394933702, 0])

# Plot
plt.figure(figsize=(12, 8))

# Solução numérica - triângulos
plt.plot(x_num_a, y_num_a, 'b^',color="hotpink", markersize=6, label='Numérico: ρ_a', alpha=0.7)

# Solução analítica - quadrados
plt.plot(x_ana_a, y_ana_a, 'bs',color="purple", markersize=6, label='Analítico: ρ_a', alpha=0.7, fillstyle='none')

plt.xlabel('Posição (z)')
plt.ylabel('Concentração mássica (ρ)')
plt.title('Comparação: Solução Numérica vs Analítica - ρ_a')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
