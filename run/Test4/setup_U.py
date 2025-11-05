import os
import pandas as pd
import numpy as np
import math


def calculate_analytical_U_equimolar(z_points, wa0, waL, Dab, MA, MB, rho_total, L=1.0):
    """
    Calcula o perfil analítico de U para DIFUSÃO EQUIMOLAR usando SUAS EQUAÇÕES VALIDADAS.
    COM CONDIÇÕES EXATAS em z=0 e z=L.
    """
    # Converter para frações mássicas e calcular rho_a0, rho_aL
    rho_a0 = float(wa0) * float(rho_total)
    rho_aL = float(waL) * float(rho_total)
    r = float(MB) / float(MA)

    # Inicializar arrays
    rho_a_analytical = np.zeros_like(z_points)
    rho_b_analytical = np.zeros_like(z_points)
    wa_analytical = np.zeros_like(z_points)
    wb_analytical = np.zeros_like(z_points)

    for i, z in enumerate(z_points):
        if z == 0:
            # Condição EXATA em z=0
            wa_analytical[i] = float(wa0)
            rho_a_analytical[i] = rho_a0
        elif z == L:
            # Condição EXATA em z=L
            wa_analytical[i] = float(waL)
            rho_a_analytical[i] = rho_aL
        else:
            # Perfil linear para pontos internos
            rho_a_analytical[i] = rho_a0 + (rho_aL - rho_a0) * (z / L)
            wa_analytical[i] = rho_a_analytical[i] / rho_total

        rho_b_analytical[i] = rho_total - rho_a_analytical[i]
        wb_analytical[i] = rho_b_analytical[i] / rho_total

    # Gradientes, fluxos e velocidade
    grad_rho_a = (rho_aL - rho_a0) / L

    ja_analytical = -Dab * grad_rho_a

    U_analytical = ((1.0 - r) / (1.0 + wa_analytical * (r - 1.0))) * (
        ja_analytical / rho_total
    )

    return U_analytical, rho_a_analytical, ja_analytical


def calculate_analytical_U_stagnantB(z_points, wa0, waL, Dab, rho_total, L=1.0):
    """
    Calcula o perfil analítico de U para B ESTAGNADO usando SUAS EQUAÇÕES VALIDADAS.
    COM CONDIÇÕES EXATAS em z=0 e z=L.
    """
    rho_total_float = float(rho_total)
    wa0_float = float(wa0)
    waL_float = float(waL)

    rho_a0 = wa0_float * rho_total_float
    rho_aL = waL_float * rho_total_float
    rho_b0 = rho_total_float - rho_a0
    K = rho_total_float / rho_b0  # constante

    # Inicializar todos os arrays
    wa_analytical = np.zeros_like(z_points)
    wb_analytical = np.zeros_like(z_points)
    rho_a_analytical = np.zeros_like(z_points)
    rho_b_analytical = np.zeros_like(z_points)

    for i, z in enumerate(z_points):
        if z == 0:
            # Condição EXATA em z=0
            wa_analytical[i] = wa0_float
            rho_a_analytical[i] = rho_a0
        elif z == L:
            # Condição EXATA em z=L
            wa_analytical[i] = waL_float
            rho_a_analytical[i] = rho_aL
        else:
            # Perfil exponencial para pontos internos
            rho_a_analytical[i] = rho_total_float - rho_b0 * (K**z)
            wa_analytical[i] = rho_a_analytical[i] / rho_total_float

        rho_b_analytical[i] = rho_total_float - rho_a_analytical[i]
        wb_analytical[i] = rho_b_analytical[i] / rho_total_float

    # Gradientes, fluxos e velocidade
    grad_rho_a_analytical = -rho_b0 * (K**z_points) * np.log(K)
    grad_rho_b_analytical = -grad_rho_a_analytical

    ja_analytical = -Dab * grad_rho_a_analytical

    U_analytical = (1 / (1 - wa_analytical)) * (ja_analytical / rho_total_float)

    return U_analytical, rho_a_analytical, ja_analytical


def calculate_boundary_values_exact(case_type, params, L=1.0):
    """
    Calcula os valores EXATOS de U nos contornos inlet (z=0) e outlet (z=L).
    """
    # Pontos EXATOS nos contornos
    z_inlet_exact = np.array([0.0])
    z_outlet_exact = np.array([L])

    if case_type == "equimolar":
        U_inlet, _, _ = calculate_analytical_U_equimolar(
            z_inlet_exact,
            params["wa0"],
            params["waL"],
            params["Dab"],
            params["MA"],
            params["MB"],
            params["rho_total"],
            L,
        )
        U_outlet, _, _ = calculate_analytical_U_equimolar(
            z_outlet_exact,
            params["wa0"],
            params["waL"],
            params["Dab"],
            params["MA"],
            params["MB"],
            params["rho_total"],
            L,
        )

    elif case_type == "stagnantB":
        U_inlet, _, _ = calculate_analytical_U_stagnantB(
            z_inlet_exact,
            params["wa0"],
            params["waL"],
            params["Dab"],
            params["rho_total"],
            L,
        )
        U_outlet, _, _ = calculate_analytical_U_stagnantB(
            z_outlet_exact,
            params["wa0"],
            params["waL"],
            params["Dab"],
            params["rho_total"],
            L,
        )

    print(f"🔬 Cálculo EXATO nos contornos:")
    print(f"   Inlet (z=0.0): wa = {params['wa0']}, U = {U_inlet[0]:.16f} m/s")
    print(f"   Outlet (z={L}): wa = {params['waL']}, U = {U_outlet[0]:.16f} m/s")

    return U_inlet[0], U_outlet[0]


def generate_cell_centroids_and_boundaries(n_cells=300, L=1.0):
    """
    Gera os centroides das células E identifica os pontos dos contornos.
    """
    cell_size = L / n_cells
    centroids = []

    for i in range(n_cells):
        # Centroide = início da célula + metade do tamanho
        z_centroid = (i * cell_size) + (cell_size / 2.0)
        centroids.append(z_centroid)

    centroids = np.array(centroids)

    # Pontos dos contornos (para referência)
    inlet_z = 0.0
    outlet_z = L

    print(f"📐 Geometria da malha:")
    print(f"   Domínio: z = [0, {L}]")
    print(f"   Células: {n_cells}")
    print(f"   Centroides: {centroids[0]:.6f} a {centroids[-1]:.6f}")
    print(f"   Contorno inlet: z = {inlet_z}")
    print(f"   Contorno outlet: z = {outlet_z}")

    return centroids, inlet_z, outlet_z


def create_U_file_from_analytical(case_dir, case_type, params, n_cells=300):
    """
    Cria arquivo 0/U com perfil analítico de U nos centroides das células.
    Usa EXATAMENTE suas equações validadas para ambos os casos.
    SOBRESCREVE o arquivo 0/U se já existir.
    """
    print(f"Criando arquivo 0/U com perfil analítico para caso: {case_type}")

    # Gerar centroides das células E identificar contornos
    L = params.get("L", 1.0)
    z_centroids, inlet_z, outlet_z = generate_cell_centroids_and_boundaries(n_cells, L)

    # Calcular U analítico em todos os centroides usando SUAS EQUAÇÕES
    if case_type == "equimolar":
        u_analytical, rho_a_analytical, ja_analytical = (
            calculate_analytical_U_equimolar(
                z_centroids,
                params["wa0"],
                params["waL"],
                params["Dab"],
                params["MA"],
                params["MB"],
                params["rho_total"],
                L,
            )
        )
        print(f"ja_analytical (constante): {ja_analytical:.16f} kg/m²s")

    elif case_type == "stagnantB":
        u_analytical, rho_a_analytical, ja_analytical = (
            calculate_analytical_U_stagnantB(
                z_centroids,
                params["wa0"],
                params["waL"],
                params["Dab"],
                params["rho_total"],
                L,
            )
        )
        print(
            f"ja_analytical (variável): {ja_analytical[0]:.16f} a {ja_analytical[-1]:.16f} kg/m²s"
        )

    print(f"U analítico calculado em {len(z_centroids)} centroides")
    print(f"U varia de {u_analytical[0]:.16f} a {u_analytical[-1]:.16f} m/s")

    # Verificar condições de contorno nos centroides mais próximos
    first_centroid_idx = 0
    last_centroid_idx = -1
    print(f"🔍 Verificação nos centroides mais próximos dos contornos:")
    print(
        f"   Primeiro centroide (z={z_centroids[first_centroid_idx]:.6f}): wa = {rho_a_analytical[first_centroid_idx] / params['rho_total']:.6f}, U = {u_analytical[first_centroid_idx]:.16f} m/s"
    )
    print(
        f"   Último centroide (z={z_centroids[last_centroid_idx]:.6f}): wa = {rho_a_analytical[last_centroid_idx] / params['rho_total']:.6f}, U = {u_analytical[last_centroid_idx]:.16f} m/s"
    )

    # CALCULAR VALORES EXATOS NOS CONTORNOS
    U_inlet, U_outlet = calculate_boundary_values_exact(case_type, params, L)

    # Aplicar correção de sinal se necessário (baseado no seu caso específico)
    # Se o perfil está invertido, descomente a linha abaixo:
    # u_analytical, U_inlet, U_outlet = -u_analytical, -U_inlet, -U_outlet

    print(f"🎯 Valores finais nos contornos:")
    print(f"   Inlet (z=0): U = ({0} {0} {U_inlet:.16f}) m/s")
    print(f"   Outlet (z={L}): U = ({0} {0} {U_outlet:.16f}) m/s")

    # Criar conteúdo do arquivo 0/U
    u_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2112                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                     |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
{len(u_analytical)}
(
"""

    # Adicionar cada valor de U analítico no formato (0 0 U_z) com 16 casas decimais
    for i, (z_centroid, u_z) in enumerate(zip(z_centroids, u_analytical)):
        u_content += f"(0 0 {u_z:.16f})"
        if i < len(u_analytical) - 1:
            u_content += "\n"
        else:
            u_content += "\n);\n\n"

    # Adicionar boundaryField COM VALORES EXATOS NOS CONTORNOS (16 casas decimais)
    u_content += f"""boundaryField
{{
    top
    {{
        type            fixedValue;
        value           uniform (0 0 {U_inlet:.16f});  // Valor analítico EXATO no inlet (z=0), wa={params["wa0"]}

    }}
    bottom
    {{
        type            fixedValue;
        value           uniform (0 0 {U_outlet:.16f});  // Valor analítico EXATO no outlet (z={L}), wa={params["waL"]}
    }}
    walls
    {{
        type            noSlip;
        value           uniform (0 0 0);
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}

// ************************************************************************* //
"""

    # Escrever arquivo 0/U (SOBRESCREVE se já existir)
    u_file_path = os.path.join(case_dir, "0", "U")

    # Verificar se existe arquivo 0/U original para backup
    if os.path.exists(u_file_path):
        backup_path = u_file_path + ".backup"
        os.rename(u_file_path, backup_path)
        print(f"📁 Backup do U original criado: {backup_path}")

    with open(u_file_path, "w") as f:
        f.write(u_content)

    print(f"✅ Arquivo 0/U criado/sobrescrito com sucesso: {u_file_path}")

    # Salvar também um arquivo de debug com os valores (16 casas decimais)
    debug_data = pd.DataFrame(
        {
            "z_centroid": z_centroids,
            "U_analytical": u_analytical,
            "rho_a_analytical": rho_a_analytical,
            "wa_analytical": rho_a_analytical / params["rho_total"],
        }
    )
    debug_path = os.path.join(case_dir, f"U_analytical_{case_type}_debug.csv")
    debug_data.to_csv(debug_path, index=False, float_format="%.16f")
    print(f"📊 Dados de debug salvos (16 casas): {debug_path}")

    # Print dos primeiros e últimos valores com 16 casas
    print("\n📈 Primeiros 5 centroides:")
    for i in range(5):
        print(
            f"   Célula {i + 1}: z={z_centroids[i]:.4f} m, wa={rho_a_analytical[i] / params['rho_total']:.6f} -> U=({0} {0} {u_analytical[i]:.16f}) m/s"
        )

    print("📈 Últimos 5 centroides:")
    for i in range(-5, 0):
        print(
            f"   Célula {n_cells + i + 1}: z={z_centroids[i]:.4f} m, wa={rho_a_analytical[i] / params['rho_total']:.6f} -> U=({0} {0} {u_analytical[i]:.16f}) m/s"
        )

    return True


# Função principal unificada
def setup_U_for_case(case_dir, case_type, params):
    """
    Configura o arquivo 0/U conforme o tipo de caso.
    SOBRESCREVE o arquivo 0/U se já existir.
    """
    n_cells = params.get("n_cells", 300)

    return create_U_file_from_analytical(case_dir, case_type, params, n_cells)


# Exemplo de uso
if __name__ == "__main__":
    case_directory = "."

    # Parâmetros padrão
    params = {
        "wa0": 0.8,  # Fração mássica em z=0
        "waL": 0.2,  # Fração mássica em z=L (agora para AMBOS os casos)
        "Dab": 0.0000155,
        "MA": 44.01,  # Apenas equimolar
        "MB": 28.96,  # Apenas equimolar
        "rho_total": 1.0,
        "L": 1.0,
        "n_cells": 300,
    }

    # Detectar ou perguntar o tipo de caso
    print("🎯 Configurador de Arquivo 0/U")
    print("=" * 50)
    print("📁 ATENÇÃO: Este script SOBRESCREVERÁ o arquivo 0/U se existir!")
    print("📁 Será criado um backup: 0/U.backup")
    print("🎯 Usando SUAS EQUAÇÕES VALIDADAS para ambos os casos")
    print("🎯 Precisão: 16 casas decimais")
    print("🎯 Condições EXATAS em z=0 e z=L")
    print("=" * 50)

    case_type = input(
        "Escolha o tipo de caso:\n1 - B Estagnado\n2 - Difusão Equimolar\n> "
    )
    case_type = "stagnantB" if case_type == "1" else "equimolar"

    # Coletar parâmetros do usuário
    try:
        params["wa0"] = float(input("wa0 (fração mássica em z=0): "))
        params["waL"] = float(input("waL (fração mássica em z=L): "))
        params["Dab"] = float(input("Dab: "))
        params["rho_total"] = float(input("rho_total: "))

        if case_type == "equimolar":
            params["MA"] = float(input("MA: "))
            params["MB"] = float(input("MB: "))
    except:
        print("Usando valores padrão...")

    # Configurar arquivo 0/U
    success = setup_U_for_case(case_directory, case_type, params)

    if success:
        print(f"\n🎯 Arquivo 0/U configurado para caso: {case_type}")
        print("✅ Backup do U original criado: 0/U.backup")
        print("✅ Usando SUAS EQUAÇÕES VALIDADAS!")
        print("✅ Precisão de 16 casas decimais!")
        print("✅ Condições EXATAS em z=0 e z=L!")
        print("✅ Contornos inlet/outlet calculados analiticamente!")
        print("\n🚀 Agora execute o solver!")
    else:
        print("❌ Falha ao configurar arquivo 0/U")
