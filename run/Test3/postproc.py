import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import numpy as np
import glob
import math


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


def calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L=1.0):
    return rho_a0 + (rho_aL - rho_a0) * z_points / L


def calculate_analytical_rho_b_equimolar(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    rho_a_analytical = calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L)
    return rho_total - rho_a_analytical


def calculate_analytical_wa_equimolar(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    rho_a_analytical = calculate_analytical_rho_a_equimolar(z_points, rho_a0, rho_aL, L)
    return rho_a_analytical / rho_total


def calculate_analytical_wb_equimolar(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    wa_analytical = calculate_analytical_wa_equimolar(
        z_points, rho_a0, rho_aL, rho_total, L
    )
    return 1 - wa_analytical


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


def calculate_analytical_U_ver_equimolar(
    z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0
):
    # U_ver = (Na + Nb) / rho_total
    Na_analytical = calculate_analytical_Na_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    Nb_analytical = calculate_analytical_Nb_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return (Na_analytical + Nb_analytical) / rho_total


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


def plot_and_export_openfoam_data(
    case_dir, field_names, set_name="myCloud", output_dir="postProcessing/sampleDict"
):
    """
    Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
    """
    print(f"Processando dados para o caso: {case_dir}")

    # Solicitar valores dos parâmetros
    try:
        rho_a0 = float(input("Digite o valor de rho_a0 (em z=0): "))
        rho_aL = float(input("Digite o valor de rho_aL (em z=1): "))
        Dab = float(input("Digite o valor de Dab: "))
        MA = float(input("Digite a massa molar MA (kg/kmol): "))
        MB = float(input("Digite a massa molar MB (kg/kmol): "))
        rho_total = float(input("Digite a densidade mássica total (rho_total): "))

        print(f"Usando rho_a0 = {rho_a0}, rho_aL = {rho_aL}, Dab = {Dab}")
        print(f"MA = {MA} kg/kmol, MB = {MB} kg/kmol, rho_total = {rho_total}")

    except ValueError:
        print("Erro: Valores inválidos. Usando valores padrão.")
        rho_a0 = 0.9
        rho_aL = 0.1
        Dab = 0.01
        MA = 28.96
        MB = 44.01
        rho_total = 1.0

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

    # DEBUG: Listar arquivos disponíveis
    print(f"\n=== DEBUG: ARQUIVOS DISPONÍVEIS ===")
    available_files = glob.glob(f"{data_path_prefix}_*.xy")
    for file in available_files:
        print(f"Arquivo encontrado: {os.path.basename(file)}")
    print("=== FIM DEBUG ARQUIVOS ===\n")

    all_data = pd.DataFrame()

    # Ler arquivo rho_rho_a_rho_b.xy para campos escalares
    rho_file = f"{data_path_prefix}_rho_rho_a_rho_b.xy"
    if os.path.exists(rho_file):
        print(f"Lendo campos escalares do arquivo: {os.path.basename(rho_file)}")
        try:
            df_rho = pd.read_csv(rho_file, sep=r"\s+", comment="#", header=None)
            print(f"DEBUG: Arquivo rho tem {df_rho.shape[1]} colunas")
            print(f"DEBUG: Primeiras 3 linhas do arquivo rho:")
            for i in range(min(3, len(df_rho))):
                print(f"  Linha {i}: {df_rho.iloc[i].tolist()}")

            if df_rho.shape[1] >= 4:
                all_data["z"] = df_rho.iloc[:, 0]  # Coluna 0 = Z
                all_data["rho"] = df_rho.iloc[:, 1]  # Coluna 1 = rho
                all_data["rho_a"] = df_rho.iloc[:, 2]  # Coluna 2 = rho_a
                all_data["rho_b"] = df_rho.iloc[:, 3]  # Coluna 3 = rho_b
                print("Campos escalares lidos com sucesso")

                # Calcular wa e wb numéricos
                all_data["wa"] = all_data["rho_a"] / all_data["rho"]
                all_data["wb"] = all_data["rho_b"] / all_data["rho"]
                print("Frações mássicas wa e wb calculadas numericamente")

            else:
                print(
                    f"ERRO: Arquivo rho tem apenas {df_rho.shape[1]} colunas, esperava pelo menos 4"
                )
                return
        except Exception as e:
            print(f"Erro ao ler arquivo rho: {e}")
            return
    else:
        print("Erro: Arquivo rho_rho_a_rho_b.xy não encontrado")
        return

    # Ler arquivo Na_Nb_U.xy para campos vetoriais
    vector_file = f"{data_path_prefix}_Na_Nb_U.xy"
    if os.path.exists(vector_file):
        print(f"Lendo campos vetoriais do arquivo: {os.path.basename(vector_file)}")
        try:
            df_vector = pd.read_csv(vector_file, sep=r"\s+", comment="#", header=None)
            print(f"DEBUG: Arquivo vetorial tem {df_vector.shape[1]} colunas")
            print(f"DEBUG: Primeiras 3 linhas do arquivo vetorial:")
            for i in range(min(3, len(df_vector))):
                print(f"  Linha {i}: {df_vector.iloc[i].tolist()}")

            if df_vector.shape[1] >= 10:
                # ESTRUTURA: z x y Naz Nax Nay Nbz Nbx Nby U
                # Colunas: 0:z, 1:x, 2:y, 3:Naz, 4:Nax, 5:Nay, 6:Nbz, 7:Nbx, 8:Nby, 9:U
                all_data["Na"] = df_vector.iloc[:, 3]  # Coluna 3 = Na_z
                all_data["Nb"] = df_vector.iloc[:, 6]  # Coluna 6 = Nb_z
                all_data["U"] = df_vector.iloc[:, 9]  # Coluna 9 = U (velocidade)
                print("Campos vetoriais (componente Z) lidos com sucesso")

                # DEBUG: Mostrar alguns valores para verificar
                print("DEBUG: Valores dos campos vetoriais (primeiros 3 pontos):")
                for i in range(min(3, len(all_data))):
                    print(
                        f"  Ponto {i}: Na={all_data['Na'].iloc[i]:.6f}, Nb={all_data['Nb'].iloc[i]:.6f}, U={all_data['U'].iloc[i]:.6f}"
                    )
            else:
                print(
                    f"ERRO: Arquivo vetorial tem apenas {df_vector.shape[1]} colunas, esperava pelo menos 10"
                )
                return
        except Exception as e:
            print(f"Erro ao ler arquivo vetorial: {e}")
            return
    else:
        print("Erro: Arquivo Na_Nb_U.xy não encontrado")

    # Calcular U_ver numericamente (Na + Nb) / rho
    if (
        "Na" in all_data.columns
        and "Nb" in all_data.columns
        and "rho" in all_data.columns
    ):
        all_data["U_ver"] = (all_data["Na"] + all_data["Nb"]) / all_data["rho"]
        print("Campo U_ver calculado numericamente: (Na + Nb) / rho")

    if all_data.empty:
        print("Nenhum dado válido foi lido para plotagem ou exportação.")
        return

    # DEBUG: Verificar dados lidos
    print(f"\n=== DEBUG: DADOS LIDOS ===")
    print(f"Colunas disponíveis: {list(all_data.columns)}")
    print(f"Número de pontos: {len(all_data)}")
    print("Primeiros 3 pontos completos:")
    for i in range(min(3, len(all_data))):
        print(f"Ponto {i}:")
        print(f"  z={all_data['z'].iloc[i]:.6f}")
        if "rho" in all_data.columns:
            print(f"  rho={all_data['rho'].iloc[i]:.6f}")
        if "rho_a" in all_data.columns:
            print(f"  rho_a={all_data['rho_a'].iloc[i]:.6f}")
        if "rho_b" in all_data.columns:
            print(f"  rho_b={all_data['rho_b'].iloc[i]:.6f}")
        if "wa" in all_data.columns:
            print(f"  wa={all_data['wa'].iloc[i]:.6f}")
        if "wb" in all_data.columns:
            print(f"  wb={all_data['wb'].iloc[i]:.6f}")
        if "Na" in all_data.columns:
            print(f"  Na={all_data['Na'].iloc[i]:.10e}")
        if "Nb" in all_data.columns:
            print(f"  Nb={all_data['Nb'].iloc[i]:.10e}")
        if "U" in all_data.columns:
            print(f"  U={all_data['U'].iloc[i]:.10e}")
        if "U_ver" in all_data.columns:
            print(f"  U_ver={all_data['U_ver'].iloc[i]:.10e}")
    print("=== FIM DEBUG DADOS ===\n")

    # Calcular soluções analíticas
    print("Calculando soluções analíticas para caso equimolar...")
    z_points = all_data["z"].values

    all_data["rho_a_analytical"] = calculate_analytical_rho_a_equimolar(
        z_points, rho_a0, rho_aL
    )
    all_data["wa_analytical"] = calculate_analytical_wa_equimolar(
        z_points, rho_a0, rho_aL, rho_total
    )
    all_data["wb_analytical"] = calculate_analytical_wb_equimolar(
        z_points, rho_a0, rho_aL, rho_total
    )
    all_data["U_analytical"] = calculate_analytical_U_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total
    )
    all_data["U_ver_analytical"] = calculate_analytical_U_ver_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total
    )
    all_data["N_a_analytical"] = calculate_analytical_Na_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total
    )
    all_data["N_b_analytical"] = calculate_analytical_Nb_equimolar(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total
    )

    print("Calculando erros...")
    if "rho_a" in all_data.columns and "rho_a_analytical" in all_data.columns:
        all_data["erro_absoluto_rho_a"] = (
            all_data["rho_a"] - all_data["rho_a_analytical"]
        )
    if "wa" in all_data.columns and "wa_analytical" in all_data.columns:
        all_data["erro_absoluto_wa"] = all_data["wa"] - all_data["wa_analytical"]
    if "wb" in all_data.columns and "wb_analytical" in all_data.columns:
        all_data["erro_absoluto_wb"] = all_data["wb"] - all_data["wb_analytical"]
    if "Na" in all_data.columns and "N_a_analytical" in all_data.columns:
        all_data["erro_absoluto_Na"] = all_data["Na"] - all_data["N_a_analytical"]
    if "Nb" in all_data.columns and "N_b_analytical" in all_data.columns:
        all_data["erro_absoluto_Nb"] = all_data["Nb"] - all_data["N_b_analytical"]
    if "U" in all_data.columns and "U_analytical" in all_data.columns:
        all_data["erro_absoluto_U"] = all_data["U"] - all_data["U_analytical"]
    if "U_ver" in all_data.columns and "U_ver_analytical" in all_data.columns:
        all_data["erro_absoluto_U_ver"] = (
            all_data["U_ver"] - all_data["U_ver_analytical"]
        )

    # Calcular diferença entre as duas velocidades (devem ser iguais)
    if "U" in all_data.columns and "U_ver" in all_data.columns:
        all_data["diff_U_U_ver"] = all_data["U"] - all_data["U_ver"]
        all_data["diff_U_U_ver_analytical"] = (
            all_data["U_analytical"] - all_data["U_ver_analytical"]
        )

    # Exportar dados para CSV com 15 casas decimais
    csv_output_path = os.path.join(case_dir, f"{set_name}_sampled_data.csv")

    # Definir colunas para exportar
    cols_to_export = ["z"]

    # Campos numéricos principais
    basic_fields = ["rho", "rho_a", "wa", "wb", "Na", "Nb", "U", "U_ver"]
    for field in basic_fields:
        if field in all_data.columns and field not in cols_to_export:
            cols_to_export.append(field)

    # Campos analíticos (apenas os necessários)
    analytical_fields_to_keep = [
        "rho_a_analytical",
        "wa_analytical",
        "wb_analytical",
        "N_a_analytical",
        "N_b_analytical",
        "U_analytical",
        "U_ver_analytical",
    ]
    for field in analytical_fields_to_keep:
        if field in all_data.columns and field not in cols_to_export:
            cols_to_export.append(field)

    # Campos de erro e diferenças
    for col in all_data.columns:
        if (
            col.startswith("erro_absoluto") or col.startswith("diff_")
        ) and col not in cols_to_export:
            cols_to_export.append(col)

    # Exportar com 15 casas decimais
    all_data[cols_to_export].to_csv(
        csv_output_path, index=False, sep=";", decimal=",", float_format="%.15e"
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
    )


def plot_from_csv(
    case_dir,
    field_names,
    set_name="myCloud",
    rho_a0=0.9,
    rho_aL=0.1,
    Dab=0.01,
    MA=28.96,
    MB=44.01,
    rho_total=1.0,
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
    try:
        data = pd.read_csv(csv_file, sep=";", decimal=",")
        print(f"Dados lidos do CSV: {len(data)} pontos")
        print(f"Colunas disponíveis: {list(data.columns)}")

        # DEBUG: Verificar as primeiras linhas
        print("DEBUG - Primeiras 3 linhas do CSV:")
        print(data.head(3))

    except Exception as e:
        print(f"Erro ao ler arquivo CSV: {e}")
        return

    # Verificar se temos a coluna Z
    if "z" not in data.columns:
        print("Erro: Coluna 'z' não encontrada no CSV")
        return

    # DEBUG: Verificar dados de velocidade
    if "U" in data.columns and "U_analytical" in data.columns:
        print(f"DEBUG - Dados de velocidade U:")
        print(f"  U numérico: min={data['U'].min():.6e}, max={data['U'].max():.6e}")
        print(
            f"  U analítico: min={data['U_analytical'].min():.6e}, max={data['U_analytical'].max():.6e}"
        )

    if "U_ver" in data.columns and "U_ver_analytical" in data.columns:
        print(f"DEBUG - Dados de velocidade U_ver:")
        print(
            f"  U_ver numérico: min={data['U_ver'].min():.6e}, max={data['U_ver'].max():.6e}"
        )
        print(
            f"  U_ver analítico: min={data['U_ver_analytical'].min():.6e}, max={data['U_ver_analytical'].max():.6e}"
        )

    # Plot 1: Concentração rho_a
    if (
        "rho_a" in data.columns
        and "rho_a_analytical" in data.columns
        and "rho_a" in field_names
    ):
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
            f"Comparação de Concentração (ρ_a) - Caso Equimolar\nρ_a0={rho_a0}, ρ_aL={rho_aL}"
        )
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(case_dir, f"{set_name}_concentration_plot.png"), dpi=300
        )
        plt.close()
        print(f"Gráfico de concentração salvo")

    # Plot 2: Frações mássicas
    if "wa" in data.columns and "wa_analytical" in data.columns and "wa" in field_names:
        plt.figure(figsize=(10, 7))
        plt.plot(
            data["z"],
            data["wa"],
            "^",
            color="coral",
            markersize=8,
            label="Numérico: w_a",
        )
        plt.plot(
            data["z"],
            data["wa_analytical"],
            "s",
            color="red",
            markersize=5,
            label="Analítico: w_a",
            fillstyle="none",
        )
        if (
            "wb" in data.columns
            and "wb_analytical" in data.columns
            and "wb" in field_names
        ):
            plt.plot(
                data["z"],
                data["wb"],
                "v",
                color="lightblue",
                markersize=8,
                label="Numérico: w_b",
            )
            plt.plot(
                data["z"],
                data["wb_analytical"],
                "s",
                color="blue",
                markersize=5,
                label="Analítico: w_b",
                fillstyle="none",
            )
        plt.xlabel("Posição (z)")
        plt.ylabel("Frações Mássicas (w_a, w_b)")
        plt.title(
            f"Comparação de Frações Mássicas - Caso Equimolar\nρ_a0={rho_a0}, ρ_aL={rho_aL}"
        )
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(case_dir, f"{set_name}_mass_fractions_plot.png"), dpi=300
        )
        plt.close()
        print(f"Gráfico de frações mássicas salvo")

    # Plot 3: Fluxos Na e Nb
    if (
        "Na" in data.columns
        and "N_a_analytical" in data.columns
        and "Na" in field_names
    ):
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
        if (
            "Nb" in data.columns
            and "N_b_analytical" in data.columns
            and "Nb" in field_names
        ):
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
        plt.ylabel("Fluxo Mássico (N)")
        plt.title(f"Comparação de Fluxos Mássicos (Na, Nb) - Caso Equimolar\nDab={Dab}")
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, f"{set_name}_flux_plot.png"), dpi=300)
        plt.close()
        print(f"Gráfico de fluxos salvo")

    # Plot 4: Velocidades U e U_ver
    if (
        "U" in data.columns
        and "U_analytical" in data.columns
        and "U" in field_names
        and "U_ver" in data.columns
        and "U_ver_analytical" in data.columns
        and "U_ver" in field_names
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Subplot 1: Velocidades numéricas e analíticas
        ax1.plot(
            data["z"],
            data["U"],
            "^",
            color="purple",
            markersize=8,
            label="Numérico: U",
        )
        ax1.plot(
            data["z"],
            data["U_analytical"],
            "s",
            color="darkviolet",
            markersize=5,
            label="Analítico: U",
            fillstyle="none",
        )
        ax1.plot(
            data["z"],
            data["U_ver"],
            "o",
            color="orange",
            markersize=6,
            label="Numérico: U_ver ((Na+Nb)/rho)",
        )
        ax1.plot(
            data["z"],
            data["U_ver_analytical"],
            "d",
            color="darkorange",
            markersize=4,
            label="Analítico: U_ver",
            fillstyle="none",
        )
        ax1.set_xlabel("Posição (z)")
        ax1.set_ylabel("Velocidade")
        ax1.set_title(f"Comparação de Velocidades - Caso Equimolar")
        ax1.legend()
        ax1.grid(True, alpha=0.4)
        ax1.set_xlim(0, 1)

        # Subplot 2: Diferença entre U e U_ver
        if "diff_U_U_ver" in data.columns:
            ax2.plot(
                data["z"],
                data["diff_U_U_ver"],
                "s-",
                color="red",
                markersize=4,
                label="Diferença Numérica: U - U_ver",
            )
            if "diff_U_U_ver_analytical" in data.columns:
                ax2.plot(
                    data["z"],
                    data["diff_U_U_ver_analytical"],
                    "o--",
                    color="darkred",
                    markersize=3,
                    label="Diferença Analítica: U - U_ver",
                )
            ax2.set_xlabel("Posição (z)")
            ax2.set_ylabel("Diferença U - U_ver")
            ax2.set_title("Diferença entre as Duas Velocidades (devem ser iguais)")
            ax2.legend()
            ax2.grid(True, alpha=0.4)
            ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(case_dir, f"{set_name}_velocity_comparison_plot.png"), dpi=300
        )
        plt.close()
        print(f"Gráfico de comparação de velocidades salvo")

    print(f"\n=== ESTATÍSTICAS DOS ERROS ===")
    if "erro_absoluto_rho_a" in data.columns and "rho_a" in field_names:
        print(
            f"rho_a - Erro Médio Absoluto: {data['erro_absoluto_rho_a'].abs().mean():.6e}"
        )
    if "erro_absoluto_wa" in data.columns and "wa" in field_names:
        print(f"wa - Erro Médio Absoluto: {data['erro_absoluto_wa'].abs().mean():.6e}")
    if "erro_absoluto_wb" in data.columns and "wb" in field_names:
        print(f"wb - Erro Médio Absoluto: {data['erro_absoluto_wb'].abs().mean():.6e}")
    if "erro_absoluto_Na" in data.columns and "Na" in field_names:
        print(f"Na - Erro Médio Absoluto: {data['erro_absoluto_Na'].abs().mean():.6e}")
    if "erro_absoluto_Nb" in data.columns and "Nb" in field_names:
        print(f"Nb - Erro Médio Absoluto: {data['erro_absoluto_Nb'].abs().mean():.6e}")
    if "erro_absoluto_U" in data.columns and "U" in field_names:
        print(f"U - Erro Médio Absoluto: {data['erro_absoluto_U'].abs().mean():.6e}")
    if "erro_absoluto_U_ver" in data.columns and "U_ver" in field_names:
        print(
            f"U_ver - Erro Médio Absoluto: {data['erro_absoluto_U_ver'].abs().mean():.6e}"
        )

    if "diff_U_U_ver" in data.columns and (
        "U" in field_names or "U_ver" in field_names
    ):
        print(f"\n=== DIFERENÇA ENTRE VELOCIDADES ===")
        print(f"U - U_ver (numérico):")
        print(f"  Média: {data['diff_U_U_ver'].mean():.6e}")
        print(f"  Desvio padrão: {data['diff_U_U_ver'].std():.6e}")
        print(f"  Máximo absoluto: {data['diff_U_U_ver'].abs().max():.6e}")

        if "diff_U_U_ver_analytical" in data.columns:
            print(f"U - U_ver (analítico):")
            print(f"  Média: {data['diff_U_U_ver_analytical'].mean():.6e}")
            print(f"  Desvio padrão: {data['diff_U_U_ver_analytical'].std():.6e}")
            print(
                f"  Máximo absoluto: {data['diff_U_U_ver_analytical'].abs().max():.6e}"
            )

    print("--- PLOTAGEM CONCLUÍDA ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Erro: Especifique pelo menos um campo para processar.")
        print("Uso: python3 postproc.py <campo1> [campo2] ...")
        print("Exemplo: python3 postproc.py rho_a")
        sys.exit(1)

    case_directory = "."
    fields_to_process = sys.argv[1:]

    print(f"Processando caso no diretório atual: {os.path.abspath(case_directory)}")
    print(f"Campos: {fields_to_process}")

    run_openfoam_postprocess(case_directory)
    plot_and_export_openfoam_data(case_directory, fields_to_process)
    print("Processo de automação completo concluído.")

    def plot_and_export_openfoam_data(
        case_dir,
        field_names,
        set_name="myCloud",
        output_dir="postProcessing/sampleDict",
    ):
        """
        Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
        """
        print(f"Processando dados para o caso: {case_dir}")

        # Solicitar valores dos parâmetros
        try:
            rho_a0 = float(input("Digite o valor de rho_a0 (em z=0): "))
            rho_aL = float(input("Digite o valor de rho_aL (em z=1): "))
            Dab = float(input("Digite o valor de Dab: "))
            MA = float(input("Digite a massa molar MA (kg/kmol): "))
            MB = float(input("Digite a massa molar MB (kg/kmol): "))
            rho_total = float(input("Digite a densidade mássica total (rho_total): "))

            print(f"DEBUG - Valores lidos:")
            print(f"  rho_a0 = {rho_a0}")
            print(f"  rho_aL = {rho_aL}")
            print(f"  Dab = {Dab}")
            print(f"  MA = {MA}")
            print(f"  MB = {MB}")
            print(f"  rho_total = {rho_total}")

        except ValueError as e:
            print(f"Erro ao ler valores: {e}")
            print("Usando valores padrão.")
            rho_a0 = 0.9
            rho_aL = 0.1
            Dab = 0.01
            MA = 28.96
            MB = 44.01
            rho_total = 1.0

        # ... resto do código permanece igual ...

    def plot_from_csv(
        case_dir,
        field_names,
        set_name="myCloud",
        rho_a0=0.9,  # REMOVER VALORES PADRÃO AQUI
        rho_aL=0.1,  # REMOVER VALORES PADRÃO AQUI
        Dab=0.01,
        MA=28.96,
        MB=44.01,
        rho_total=1.0,
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
        try:
            data = pd.read_csv(csv_file, sep=";", decimal=",")
            print(f"Dados lidos do CSV: {len(data)} pontos")
            print(f"Colunas disponíveis: {list(data.columns)}")

            # DEBUG: Verificar as primeiras linhas
            print("DEBUG - Primeiras 3 linhas do CSV:")
            print(data.head(3))

        except Exception as e:
            print(f"Erro ao ler arquivo CSV: {e}")
            return

        # DEBUG: Mostrar valores que estão sendo usados
        print(f"DEBUG - Parâmetros usados para plotagem:")
        print(f"  rho_a0 = {rho_a0}")
        print(f"  rho_aL = {rho_aL}")
        print(f"  Dab = {Dab}")
        print(f"  MA = {MA}")
        print(f"  MB = {MB}")
        print(f"  rho_total = {rho_total}")

        # ... resto do código permanece igual ...
