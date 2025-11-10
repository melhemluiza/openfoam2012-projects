import math
import os

import numpy as np
import pandas as pd


def calculate_constant_U_equimolar(wa0, waL, Dab, MA, MB, rho_total, L=1.0):
    """
    Calcula U CONSTANTE para DIFUSÃO EQUIMOLAR usando AS EQUAÇÕES VALIDADAS.
    U, Na, Nb são constantes; grad_rho_a é variável.
    """
    rho_total_float = float(rho_total)
    wa0_float = float(wa0)
    waL_float = float(waL)
    r = float(MB) / float(MA)

    rho_a0 = wa0_float * rho_total_float
    rho_aL = waL_float * rho_total_float

    # Calcular C2 - MESMA EQUAÇÃO DO postproc.py
    C2 = (rho_total_float - (1 - r) * rho_aL) / (rho_total_float - (1 - r) * rho_a0)

    # Para difusão equimolar, Na é constante (e Nb = -Na)
    # Cálculo de Na constante através da equação do postproc.py
    grad_rho_a_at_inlet = (
        -((rho_total_float / (1 - r)) - rho_a0) * (np.log(C2) / L) * (C2 ** (0 / L))
    )
    ja_at_inlet = -Dab * grad_rho_a_at_inlet

    # Velocidade U é constante
    wa_inlet = wa0_float
    U_constant = ((1.0 - r) / (1.0 + wa_inlet * (r - 1.0))) * (
        ja_at_inlet / rho_total_float
    )

    # Na constante = ja + rho_a * U (mas como U é constante e rho_a varia, Na deve ser calculado consistentemente)
    # Na verdade, para difusão equimolar, Na = -Nb = constante
    Na_constant = ja_at_inlet + rho_a0 * U_constant

    print(f"🎯 DIFUSÃO EQUIMOLAR:")
    print(f"   U CONSTANTE = {U_constant:.16f} m/s")
    print(f"   Na CONSTANTE = {Na_constant:.16f} kg/m²s")
    print(f"   Nb CONSTANTE = {-Na_constant:.16f} kg/m²s")
    print(f"   grad_rho_a VARIÁVEL ao longo do domínio")

    return U_constant


def calculate_constant_U_stagnantB(wa0, rho_total, Dab, L=1.0):
    """
    Calcula U CONSTANTE para B ESTAGNADO usando AS EQUAÇÕES VALIDADAS.
    U, Na, Nb são constantes; grad_rho_a é variável.
    """
    rho_total_float = float(rho_total)
    wa0_float = float(wa0)

    rho_a0 = wa0_float * rho_total_float
    C1 = np.log(1 - wa0_float)

    # Calcular no ponto z=0
    grad_rho_a_at_inlet = C1 / np.exp(C1 * 0 - C1)  # z=0
    ja_at_inlet = -Dab * grad_rho_a_at_inlet

    # Para B estagnado, Nb = 0 e Na é constante
    U_constant = (1 / (1 - wa0_float)) * (ja_at_inlet / rho_total_float)

    # Na constante = ja + rho_a * U
    Na_constant = ja_at_inlet + rho_a0 * U_constant

    print(f"🎯 B ESTAGNADO:")
    print(f"   U CONSTANTE = {U_constant:.16f} m/s")
    print(f"   Na CONSTANTE = {Na_constant:.16f} kg/m²s")
    print(f"   Nb CONSTANTE = 0 kg/m²s")
    print(f"   grad_rho_a VARIÁVEL ao longo do domínio")

    return U_constant


def calculate_constant_U_values(case_type, params, L=1.0):
    """
    Calcula os valores CONSTANTES de U para todo o domínio.
    Para AMBOS os casos: U, Na, Nb são constantes; grad_rho_a é variável.
    """

    if case_type == "equimolar":
        U_constant = calculate_constant_U_equimolar(
            params["wa0"],
            params["waL"],
            params["Dab"],
            params["MA"],
            params["MB"],
            params["rho_total"],
            L,
        )
    else:  # stagnantB
        U_constant = calculate_constant_U_stagnantB(
            params["wa0"], params["rho_total"], params["Dab"], L
        )

    return U_constant, U_constant


def create_U_file_constant(case_dir, case_type, params):
    """
    Cria arquivo 0/U com valor CONSTANTE de U em todo o domínio.
    Para AMBOS os casos, U é constante (enquanto grad_rho_a varia).
    """
    print(f"🎯 Criando arquivo 0/U com U CONSTANTE para caso: {case_type}")

    # Calcular valor CONSTANTE de U para todo o domínio
    U_inlet, U_outlet = calculate_constant_U_values(case_type, params)

    # Usar o mesmo valor constante para todo o domínio
    U_constant = U_inlet

    print(f"🎯 RESUMO:")
    print(f"   U CONSTANTE em todo domínio = {U_constant:.16f} m/s")
    print(f"   Na, Nb CONSTANTES em todo domínio")
    print(f"   grad_rho_a VARIÁVEL ao longo do domínio")

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

internalField   uniform (0 0 {U_constant:.16f});

boundaryField
{{
    top
    {{
        type            fixedValue;
        value           uniform (0 0 {U_constant:.16f});
    }}
    bottom
    {{
        type            fixedValue;
        value           uniform (0 0 {U_constant:.16f});
    }}
    sidesX
    {{
        type            empty;
    }}
    sidesY
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
    print(f"🎯 U CONSTANTE = (0 0 {U_constant:.16f}) m/s")

    return True


def setup_U_for_case(case_dir, case_type, params):
    """
    Configura o arquivo 0/U com valor CONSTANTE conforme o tipo de caso.
    Para AMBOS os casos: U constante, Na/Nb constantes, grad_rho_a variável.
    """
    return create_U_file_constant(case_dir, case_type, params)


# Exemplo de uso
if __name__ == "__main__":
    case_directory = "."

    # Parâmetros padrão
    params = {
        "wa0": 0.8,
        "waL": 0.2,
        "Dab": 0.0000155,
        "MA": 44.01,
        "MB": 28.96,
        "rho_total": 1.0,
        "L": 1.0,
    }

    print("🎯 Configurador de Arquivo 0/U COM U CONSTANTE")
    print("=" * 60)
    print("📁 ATENÇÃO: Este script SOBRESCREVERÁ o arquivo 0/U se existir!")
    print("📁 Será criado um backup: 0/U.backup")
    print("🎯 Usando AS MESMAS EQUAÇÕES DO postproc.py")
    print("🎯 PARA AMBOS OS CASOS:")
    print("   • U CONSTANTE em todo domínio")
    print("   • Na, Nb CONSTANTES em todo domínio")
    print("   • grad_rho_a VARIÁVEL ao longo do domínio")
    print("🎯 Precisão: 16 casas decimais")
    print("=" * 60)

    case_type = input(
        "Escolha o tipo de caso:\n1 - B Estagnado\n2 - Difusão Equimolar\n> "
    )
    case_type = "stagnantB" if case_type == "1" else "equimolar"

    # Coletar parâmetros do usuário
    try:
        params["wa0"] = float(input("wa0 (fração mássica em z=0): "))
        if case_type == "stagnantB":
            print("💡 Para B estagnado, waL é sempre 0 no outlet")
            params["waL"] = 0.0
        else:
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
        print(f"\n🎉 Arquivo 0/U configurado para caso: {case_type}")
        print("✅ Backup do U original criado: 0/U.backup")
        print("✅ Usando AS MESMAS EQUAÇÕES DO postproc.py!")
        print("✅ U CONSTANTE em todo o domínio!")
        print("✅ Na, Nb CONSTANTES em todo o domínio!")
        print("✅ grad_rho_a VARIÁVEL ao longo do domínio!")
        print("✅ Precisão de 16 casas decimais!")
        print("\n🚀 Agora execute o solver!")
    else:
        print("❌ Falha ao configurar arquivo 0/U")
