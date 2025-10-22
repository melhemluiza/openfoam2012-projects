import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import numpy as np
import glob
import math

# --- Funções Analíticas para Difusão Equimolar ---


def calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L=1.0):
    return rho_a0 + (rho_aL - rho_a0) * z_points / L


def calculate_analytical_rho_b_equimolar(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    rho_a_analytical = calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L)
    return rho_total - rho_a_analytical


def calculate_analytical_wa_equimolar(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    rho_a_analytical = calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L)
    return rho_a_analytical / rho_total


def calculate_analytical_ja_equimolar(Dab, rho_a0, rho_aL, L=1.0):
    grad_rho_a = (rho_aL - rho_a0) / L
    return -Dab * grad_rho_a


def calculate_analytical_jb_equimolar(Dab, rho_a0, rho_aL, L=1.0):
    return -calculate_analytical_ja_equimolar(Dab, rho_a0, rho_aL, L)


def calculate_analytical_U_equimolar(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    ratio = MB / MA
    grad_rho_a = (rho_aL - rho_a0) / L
    wa_analytical = calculate_analytical_wa_equimolar(
        z_points, rho_a0, rho_aL, rho_total, L
    )

    epsilon = 1e-12
    denominator = 1 - wa_analytical * (1 - ratio)
    denominator[np.abs(denominator) < epsilon] = epsilon

    U_analytical = (-Dab / rho_total) * grad_rho_a * (1 - ratio) / denominator
    return U_analytical


def calculate_analytical_Na_equimolar(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    ja_analytical = calculate_analytical_ja_equimolar(Dab, rho_a0, rho_aL, L)
    rho_a_analytical = calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L)
    U_analytical = calculate_analytical_U_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return ja_analytical + rho_a_analytical * U_analytical


def calculate_analytical_Nb_equimolar(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    jb_analytical = calculate_analytical_jb_equimolar(Dab, rho_a0, rho_aL, L)
    rho_b_analytical = calculate_analytical_rho_b_equimolar(
        z_points, rho_a0, rho_aL, rho_total, L
    )
    U_analytical = calculate_analytical_U_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return jb_analytical + rho_b_analytical * U_analytical


# --- Funções Analíticas para Difusão com B Estagnado ---


def calculate_analytical_rho_a_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    w_a0 = rho_a0 / rho_total
    w_aL = rho_aL / rho_total

    # Evitar divisão por zero ou logaritmo de zero/negativo
    if (1 - w_a0) <= 0 or (1 - w_aL) <= 0:
        print(
            "Aviso: Condições de contorno podem levar a logaritmo inválido para B estagnado."
        )
        return np.full_like(z_points, np.nan)

    term_base = (1 - w_aL) / (1 - w_a0)
    exponent = z_points / L

    w_a_analytical = 1 - (1 - w_a0) * (term_base**exponent)
    return w_a_analytical * rho_total


def calculate_analytical_rho_b_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    rho_a_analytical = calculate_analytical_rho_a_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return rho_total - rho_a_analytical


def calculate_analytical_wa_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    return (
        calculate_analytical_rho_a_stagnantB(
            z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
        )
        / rho_total
    )


def calculate_analytical_Na_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    w_a0 = rho_a0 / rho_total
    w_aL = rho_aL / rho_total

    if (1 - w_a0) <= 0 or (1 - w_aL) <= 0:
        return np.full_like(z_points, np.nan)

    Na_analytical = (rho_total * Dab / L) * math.log((1 - w_aL) / (1 - w_a0))
    return np.full_like(z_points, Na_analytical)


def calculate_analytical_Nb_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    return np.zeros_like(z_points)


def calculate_analytical_U_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    Na_analytical = calculate_analytical_Na_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return Na_analytical / rho_total


def calculate_analytical_ja_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    Na_analytical = calculate_analytical_Na_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    wa_analytical = calculate_analytical_wa_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return Na_analytical * (1 - wa_analytical)


def calculate_analytical_jb_stagnantB(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    Na_analytical = calculate_analytical_Na_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    wa_analytical = calculate_analytical_wa_stagnantB(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return -Na_analytical * wa_analytical  # jb = -Na * wa


# --- Funções de Leitura e Pós-processamento ---


def run_openfoam_postprocess(case_dir):
    """
    Executa o comando postProcess do OpenFOAM.
    """
    print("Iniciando o pós-processamento do OpenFOAM...")
    try:
        result = subprocess.run(
            ["postProcess", "-func", "sampleDict", "-latestTime"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        print("Pós-processamento do OpenFOAM concluído com sucesso.")
        if result.stderr:
            print("Avisos do postProcess:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Erro durante o pós-processamento do OpenFOAM: {e}")
        print(f"Saída de erro: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "Erro: Comando 'postProcess' não encontrado. Verifique o ambiente OpenFOAM."
        )
        sys.exit(1)


def plot_and_export_openfoam_data(
    case_dir, field_names, set_name="myCloud", output_dir="postProcessing/sampleDict"
):
    """
    Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
    """
    print(f"Processando dados para o caso: {case_dir}")

    # Solicitar valores das condições de contorno e parâmetros
    try:
        rho_a0 = float(input("Digite o valor de rho_a0 (em z=0): "))
        rho_aL = float(input("Digite o valor de rho_aL (em z=1): "))
        Dab = float(input("Digite o valor de Dab: "))
        MA = float(input("Digite a massa molar MA (kg/kmol): "))
        MB = float(input("Digite a massa molar MB (kg/kmol): "))
        rho_total = float(
            input("Digite a densidade mássica total (rho_total, assumida constante): ")
        )

        print(f"Usando rho_a0 = {rho_a0}, rho_aL = {rho_aL}, Dab = {Dab}")
        print(f"MA = {MA} kg/kmol, MB = {MB} kg/kmol, rho_total = {rho_total}")

    except ValueError:
        print("Erro: Valores inválidos. Usando valores padrão.")
        rho_a0 = 0.5
        rho_aL = 0.0
        Dab = 0.01
        MA = 28.96
        MB = 44.01
        rho_total = 1.0

    # PERGUNTAR O TIPO DE CASO
    case_type = None
    while case_type not in ["equimolar", "stagnantB"]:
        case_type = (
            input("Digite o tipo de caso ('equimolar' ou 'stagnantB'): ")
            .strip()
            .lower()
        )
        if case_type not in ["equimolar", "stagnantb"]:
            print("Entrada inválida. Por favor, digite 'equimolar' ou 'stagnantB'.")
        elif case_type == "stagnantb":
            case_type = "stagnantB"

    print(f"Usando caso: {case_type}")

    post_processing_path = os.path.join(case_dir, output_dir)
    if not os.path.exists(post_processing_path):
        print(
            f"Erro: Diretório de pós-processamento não encontrado em {post_processing_path}"
        )
        return

    time_dirs = [
        d
        for d in os.listdir(post_processing_path)
        if os.path.isdir(os.path.join(post_processing_path, d))
    ]
    if not time_dirs:
        print(f"Erro: Nenhum diretório de tempo encontrado em {post_processing_path}")
        return

    latest_time = sorted(time_dirs, key=float)[-1]
    data_path_prefix = os.path.join(post_processing_path, latest_time, set_name)

    all_data = pd.DataFrame()

    # Primeiro: tentar ler coordenadas do arquivo rho_rho_a_rho_b.xy
    rho_file = f"{data_path_prefix}_rho_rho_a_rho_b.xy"
    if os.path.exists(rho_file):
        print(f"Lendo coordenadas do arquivo: {os.path.basename(rho_file)}")
        try:
            df_rho = pd.read_csv(rho_file, sep=r"\s+", comment="#", header=None)
            # Estrutura: Z, rho, rho_a, rho_b
            if df_rho.shape[1] >= 4:
                all_data["z"] = df_rho.iloc[:, 0]  # Coluna 0 = Z
                all_data["x"] = np.zeros(len(df_rho))  # Assumindo x=0 para 1D
                all_data["y"] = np.zeros(len(df_rho))  # Assumindo y=0 para 1D
                print("Coordenadas lidas do arquivo rho_rho_a_rho_b.xy")

                # Ler campos do arquivo rho
                if "rho" in field_names:
                    all_data["rho"] = df_rho.iloc[:, 1]  # Coluna 1 = rho
                    print("Campo rho lido do arquivo")
                if "rho_a" in field_names:
                    all_data["rho_a"] = df_rho.iloc[:, 2]  # Coluna 2 = rho_a
                    print("Campo rho_a lido do arquivo")
                if "rho_b" in field_names:
                    all_data["rho_b"] = df_rho.iloc[:, 3]  # Coluna 3 = rho_b
                    print("Campo rho_b lido do arquivo")

        except Exception as e:
            print(f"Erro ao ler coordenadas do arquivo rho: {e}")
            return
    else:
        print(f"Erro: Arquivo {rho_file} não encontrado")
        return

    # Segundo: ler campos do arquivo Na_Nb_U.xy (10 colunas)
    flux_file = f"{data_path_prefix}_Na_Nb_U.xy"
    if os.path.exists(flux_file):
        print(f"Lendo campos do arquivo: {os.path.basename(flux_file)}")
        try:
            df_flux = pd.read_csv(flux_file, sep=r"\s+", comment="#", header=None)
            # ESTRUTURA CORRIGIDA: z x y Naz Nax Nay Nbz Nbx Nby U
            # Colunas: 0:z, 1:x, 2:y, 3:Naz, 4:Nax, 5:Nay, 6:Nbz, 7:Nbx, 8:Nby, 9:U
            if df_flux.shape[1] >= 10:
                if "Na" in field_names and "Na" not in all_data.columns:
                    all_data["Na"] = df_flux.iloc[
                        :, 3
                    ]  # Coluna 3 = Naz (componente z de Na)
                    print("Campo Na lido do arquivo de fluxos (coluna 3)")
                if "Nb" in field_names and "Nb" not in all_data.columns:
                    all_data["Nb"] = df_flux.iloc[
                        :, 6
                    ]  # Coluna 6 = Nbz (componente z de Nb)
                    print("Campo Nb lido do arquivo de fluxos (coluna 6)")
                if "U" in field_names and "U" not in all_data.columns:
                    all_data["U"] = df_flux.iloc[:, 9]  # Coluna 9 = U
                    print("Campo U lido do arquivo de fluxos (coluna 9)")

                # DEBUG: Verificar valores lidos
                if "Na" in all_data.columns:
                    print(
                        f"  Na - min: {all_data['Na'].min():.6e}, max: {all_data['Na'].max():.6e}"
                    )
                if "Nb" in all_data.columns:
                    print(
                        f"  Nb - min: {all_data['Nb'].min():.6e}, max: {all_data['Nb'].max():.6e}"
                    )
                if "U" in all_data.columns:
                    print(
                        f"  U - min: {all_data['U'].min():.6e}, max: {all_data['U'].max():.6e}"
                    )

            else:
                print(f"Aviso: Arquivo de fluxos tem apenas {df_flux.shape[1]} colunas")
        except Exception as e:
            print(f"Erro ao ler arquivo de fluxos: {e}")
    else:
        print(f"Aviso: Arquivo {flux_file} não encontrado")

    if all_data.empty:
        print("Nenhum dado válido foi lido para plotagem ou exportação.")
        return

    print(f"Dados lidos com sucesso: {len(all_data)} pontos")
    print(f"Colunas disponíveis: {list(all_data.columns)}")

    # Calcular soluções analíticas baseadas no tipo de caso escolhido
    print(f"Calculando soluções analíticas para o caso: {case_type}...")
    if case_type == "equimolar":
        all_data["rho_a_analytical"] = calculate_analytical_rho_a_equimolar(
            all_data["z"].values, rho_a0, rho_aL
        )
        all_data["rho_b_analytical"] = calculate_analytical_rho_b_equimolar(
            all_data["z"].values, rho_a0, rho_aL, rho_total
        )
        all_data["wa_analytical"] = calculate_analytical_wa_equimolar(
            all_data["z"].values, rho_a0, rho_aL, rho_total
        )
        all_data["ja_analytical"] = calculate_analytical_ja_equimolar(
            Dab, rho_a0, rho_aL
        )
        all_data["jb_analytical"] = calculate_analytical_jb_equimolar(
            Dab, rho_a0, rho_aL
        )
        all_data["U_analytical"] = calculate_analytical_U_equimolar(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["N_a_analytical"] = calculate_analytical_Na_equimolar(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["N_b_analytical"] = calculate_analytical_Nb_equimolar(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
    elif case_type == "stagnantB":
        all_data["rho_a_analytical"] = calculate_analytical_rho_a_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["rho_b_analytical"] = calculate_analytical_rho_b_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["wa_analytical"] = calculate_analytical_wa_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["ja_analytical"] = calculate_analytical_ja_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["jb_analytical"] = calculate_analytical_jb_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["U_analytical"] = calculate_analytical_U_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["N_a_analytical"] = calculate_analytical_Na_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )
        all_data["N_b_analytical"] = calculate_analytical_Nb_stagnantB(
            all_data["z"].values, rho_a0, rho_aL, Dab, MA, MB, rho_total
        )

    print("Calculando erros...")
    if "rho_a" in all_data.columns and "rho_a_analytical" in all_data.columns:
        all_data["erro_absoluto_rho_a"] = (
            all_data["rho_a"] - all_data["rho_a_analytical"]
        )
    if "Na" in all_data.columns and "N_a_analytical" in all_data.columns:
        all_data["erro_absoluto_Na"] = all_data["Na"] - all_data["N_a_analytical"]
    if "Nb" in all_data.columns and "N_b_analytical" in all_data.columns:
        all_data["erro_absoluto_Nb"] = all_data["Nb"] - all_data["N_b_analytical"]
    if "U" in all_data.columns and "U_analytical" in all_data.columns:
        all_data["erro_absoluto_U"] = all_data["U"] - all_data["U_analytical"]

    # Exportar dados para CSV com 15 casas decimais
    csv_output_path = os.path.join(case_dir, f"{set_name}_sampled_data.csv")

    # Definir colunas para exportar
    cols_to_export = ["z", "x", "y"]
    for field in field_names:
        if field in all_data.columns and field not in cols_to_export:
            cols_to_export.append(field)
    for col in all_data.columns:
        if col.endswith("_analytical") or col.startswith("erro_absoluto"):
            if col not in cols_to_export:
                cols_to_export.append(col)

    # Exportar com 15 casas decimais
    all_data[cols_to_export].to_csv(
        csv_output_path, index=False, sep=";", decimal=",", float_format="%.15f"
    )
    print(f"Dados exportados para CSV (15 casas decimais): {csv_output_path}")

    # Plotar gráficos
    plot_from_csv(
        case_dir,
        field_names,
        set_name,
        rho_a0,
        rho_aL,
        Dab,
        MA,
        MB,
        rho_total,
        case_type,
    )


def plot_from_csv(
    case_dir,
    field_names,
    set_name="myCloud",
    rho_a0=0.5,
    rho_aL=0.0,
    Dab=0.01,
    MA=28.96,
    MB=44.01,
    rho_total=1.0,
    case_type="stagnantB",
):
    """
    Lê os dados do CSV gerado e plota comparando com analítico
    """
    print(f"\n--- PLOTANDO DO CSV ---")

    csv_file = os.path.join(case_dir, f"{set_name}_sampled_data.csv")

    if not os.path.exists(csv_file):
        print(f"Erro: Arquivo CSV não encontrado: {csv_file}")
        return

    # Ler dados do CSV
    data = pd.read_csv(csv_file, sep=";", decimal=",")
    print(f"Dados lidos do CSV: {len(data)} pontos")
    print(f"Colunas disponíveis: {list(data.columns)}")

    # Verificar se temos a coluna Z
    if "z" not in data.columns:
        print("Erro: Coluna 'z' não encontrada no CSV")
        return

    # Plot 1: Concentração
    if "rho_a" in data.columns and "rho_a_analytical" in data.columns:
        plt.figure(figsize=(10, 7))
        plt.plot(
            data["z"],
            data["rho_a"],
            "^",
            color="hotpink",
            markersize=8,
            label="Numérico: ρ_a",
        )
        plt.plot(
            data["z"],
            data["rho_a_analytical"],
            "s",
            color="blue",
            markersize=5,
            label="Analítico: ρ_a",
            fillstyle="none",
        )
        plt.xlabel("Posição (z)")
        plt.ylabel("Concentração mássica (ρ_a)")
        plt.title(
            f"Comparação de Concentração (ρ_a) - Caso: {case_type}\nρ_a0={rho_a0}, ρ_aL={rho_aL}"
        )
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(case_dir, f"{set_name}_concentration_plot.png"), dpi=300
        )
        plt.close()
        print(
            f"Gráfico de concentração salvo: {os.path.join(case_dir, f'{set_name}_concentration_plot.png')}"
        )

    # Plot 2: Fluxos
    if "Na" in data.columns and "N_a_analytical" in data.columns:
        plt.figure(figsize=(10, 7))
        plt.plot(
            data["z"],
            data["Na"],
            "^",
            color="green",
            markersize=8,
            label="Numérico: Na",
        )
        plt.plot(
            data["z"],
            data["N_a_analytical"],
            "s",
            color="darkgreen",
            markersize=5,
            label="Analítico: Na",
            fillstyle="none",
        )
        if "Nb" in data.columns and "N_b_analytical" in data.columns:
            plt.plot(
                data["z"],
                data["Nb"],
                "v",
                color="red",
                markersize=8,
                label="Numérico: Nb",
            )
            plt.plot(
                data["z"],
                data["N_b_analytical"],
                "s",
                color="darkred",
                markersize=5,
                label="Analítico: Nb",
                fillstyle="none",
            )
        plt.xlabel("Posição (z)")
        plt.ylabel("Fluxo Mássico Total (N)")
        plt.title(
            f"Comparação de Fluxos Mássicos (Na, Nb) - Caso: {case_type}\nDab={Dab}, MA={MA}, MB={MB}"
        )
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"{set_name}_flux_plot.png"), dpi=300)
        plt.close()
        print(
            f"Gráfico de fluxos salvo: {os.path.join(case_dir, f'{set_name}_flux_plot.png')}"
        )

        # Plot 3: Velocidade - CORRIGIDO
        if "U" in data.columns and "U_analytical" in data.columns:
            plt.figure(figsize=(10, 7))

            # DEBUG: Verificar o que está sendo plotado
            print(f"Plotando U - Dados usados:")
            print(f"  z range: {data['z'].min():.3f} a {data['z'].max():.3f}")
            print(f"  U range: {data['U'].min():.6e} a {data['U'].max():.6e}")
            print(
                f"  U_analytical range: {data['U_analytical'].min():.6e} a {data['U_analytical'].max():.6e}"
            )

            # Plotar U numérico e analítico CORRETAMENTE
            plt.plot(
                data["z"],
                data["U"],
                "^",
                color="purple",
                markersize=8,
                label="Numérico: U",
            )
            plt.plot(
                data["z"],
                data["U_analytical"],
                "s",
                color="darkviolet",
                markersize=5,
                label="Analítico: U",
                fillstyle="none",
            )

            plt.xlabel("Posição (z)")
            plt.ylabel("Velocidade (U)")
            plt.title(f"Comparação de Velocidade (U) - Caso: {case_type}")
            plt.legend()
            plt.grid(True, alpha=0.4)
            plt.xlim(0, 1)
            plt.tight_layout()

            plt.savefig(
                os.path.join(case_dir, f"{set_name}_velocity_plot.png"), dpi=300
            )
            plt.close()
            print(
                f"Gráfico de velocidade salvo: {os.path.join(case_dir, f'{set_name}_velocity_plot.png')}"
            )

        print(f"\n=== ESTATÍSTICAS DOS ERROS ===")
        if "erro_absoluto_rho_a" in data.columns:
            print(
                f"rho_a - Erro Médio Absoluto: {data['erro_absoluto_rho_a'].abs().mean():.6e}"
            )
        if "erro_absoluto_Na" in data.columns:
            print(
                f"Na - Erro Médio Absoluto: {data['erro_absoluto_Na'].abs().mean():.6e}"
            )
        if "erro_absoluto_Nb" in data.columns:
            print(
                f"Nb - Erro Médio Absoluto: {data['erro_absoluto_Nb'].abs().mean():.6e}"
            )
        if "erro_absoluto_U" in data.columns:
            print(
                f"U - Erro Médio Absoluto: {data['erro_absoluto_U'].abs().mean():.6e}"
            )

        print("--- PLOTAGEM CONCLUÍDA ---")

    print(f"\n=== ESTATÍSTICAS DOS ERROS ===")
    if "erro_absoluto_rho_a" in data.columns:
        print(
            f"rho_a - Erro Médio Absoluto: {data['erro_absoluto_rho_a'].abs().mean():.6e}"
        )
    if "erro_absoluto_Na" in data.columns:
        print(f"Na - Erro Médio Absoluto: {data['erro_absoluto_Na'].abs().mean():.6e}")
    if "erro_absoluto_Nb" in data.columns:
        print(f"Nb - Erro Médio Absoluto: {data['erro_absoluto_Nb'].abs().mean():.6e}")
    if "erro_absoluto_U" in data.columns:
        print(f"U - Erro Médio Absoluto: {data['erro_absoluto_U'].abs().mean():.6e}")

    print("--- PLOTAGEM CONCLUÍDA ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Erro: Especifique pelo menos um campo para processar.")
        print("Uso: python3 postproc.py <campo1> [campo2] ...")
        print("Exemplo: python3 postproc.py rho_a Na Nb U")
        sys.exit(1)

    case_directory = "."
    fields_to_process = sys.argv[1:]

    print(f"Processando caso no diretório atual: {os.path.abspath(case_directory)}")
    print(f"Campos: {fields_to_process}")

    run_openfoam_postprocess(case_directory)
    plot_and_export_openfoam_data(case_directory, fields_to_process)
    print("Processo de automação completo concluído.")
