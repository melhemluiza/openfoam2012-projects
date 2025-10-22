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


def calculate_analytical_rho_a(z_points, rho_a0, rho_aL, L=1.0):
    """
    Calcula a solução analítica para a concentração mássica de A (rho_a).
    rho_a(z) = rho_a0 + (rho_aL - rho_a0) * z / L
    """
    return rho_a0 + (rho_aL - rho_a0) * z_points / L


def calculate_analytical_rho_b(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    """
    Calcula a solução analítica para a concentração mássica de B (rho_b).
    rho_b(z) = rho_total - rho_a(z)
    """
    rho_a_analytical = calculate_analytical_rho_a(z_points, rho_a0, rho_aL, L)
    return rho_total - rho_a_analytical


def calculate_analytical_wa(z_points, rho_a0, rho_aL, rho_total, L=1.0):
    """
    Calcula a fração mássica de A (wa).
    wa(z) = rho_a(z) / rho_total
    """
    rho_a_analytical = calculate_analytical_rho_a(z_points, rho_a0, rho_aL, L)
    return rho_a_analytical / rho_total


def calculate_analytical_ja(Dab, rho_a0, rho_aL, L=1.0):
    """
    Calcula o fluxo difusivo mássico de A (ja).
    ja = -Dab * d(rho_a)/dz
    """
    grad_rho_a = (rho_aL - rho_a0) / L
    return -Dab * grad_rho_a


def calculate_analytical_jb(Dab, rho_a0, rho_aL, L=1.0):
    """
    Calcula o fluxo difusivo mássico de B (jb).
    jb = -ja
    """
    return -calculate_analytical_ja(Dab, rho_a0, rho_aL, L)


def calculate_analytical_v(z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0):
    """
    Calcula a velocidade média mássica (v).
    v(z) = (-Dab / rho_total) * (grad_rho_a) * (1 - ratio) / (1 - wa(z) * (1 - ratio))
    onde ratio = MB / MA
    """
    ratio = MB / MA
    grad_rho_a = (rho_aL - rho_a0) / L
    wa_analytical = calculate_analytical_wa(z_points, rho_a0, rho_aL, rho_total, L)

    # Evitar divisão por zero se o denominador for muito próximo de zero
    denominator = 1 - wa_analytical * (1 - ratio)
    # Adicionar uma pequena constante para evitar divisão por zero exata
    epsilon = 1e-12
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)

    v_analytical = (-Dab / rho_total) * grad_rho_a * (1 - ratio) / denominator
    return v_analytical


def calculate_analytical_Na(z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0):
    """
    Calcula o fluxo mássico total de A (Na).
    Na(z) = ja + rho_a(z) * v(z)
    """
    ja_analytical = calculate_analytical_ja(Dab, rho_a0, rho_aL, L)
    rho_a_analytical = calculate_analytical_rho_a(z_points, rho_a0, rho_aL, L)
    v_analytical = calculate_analytical_v(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return ja_analytical + rho_a_analytical * v_analytical


def calculate_analytical_Nb(z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L=1.0):
    """
    Calcula o fluxo mássico total de B (Nb).
    Nb(z) = jb + rho_b(z) * v(z)
    """
    jb_analytical = calculate_analytical_jb(Dab, rho_a0, rho_aL, L)
    rho_b_analytical = calculate_analytical_rho_b(
        z_points, rho_a0, rho_aL, rho_total, L
    )
    v_analytical = calculate_analytical_v(
        z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
    )
    return jb_analytical + rho_b_analytical * v_analytical


def calculate_analytical_solutions(z_points, rho_a0, rho_aL, L, Dab, MA, MB, rho_total):
    """
    Calcula todas as soluções analíticas usando as funções validadas.
    """
    return {
        "rho_a_analytical": calculate_analytical_rho_a(z_points, rho_a0, rho_aL, L),
        "Na_analytical": calculate_analytical_Na(
            z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
        ),
        "Nb_analytical": calculate_analytical_Nb(
            z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
        ),
        "v_analytical": calculate_analytical_v(
            z_points, rho_a0, rho_aL, Dab, MA, MB, rho_total, L
        ),
    }


def plot_and_export_openfoam_data(
    case_dir, field_names, set_name="myCloud", output_dir="postProcessing/sampleDict"
):
    """
    Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
    Específico para equimolarDiffusionFoam.
    """
    print(f"Processando dados para o caso: {case_dir}")

    # Solicitar parâmetros do usuário para cálculo da solução analítica
    try:
        rho_a0 = float(input("Digite o valor de rho_a0 (concentração de A em z=0): "))
        rho_aL = float(input("Digite o valor de rho_aL (concentração de A em z=L): "))
        L = 1.0  # Fixo em 1
        Dab = float(input("Digite o coeficiente de difusão Dab: "))
        MA = float(input("Digite o peso molecular MA: "))
        MB = float(input("Digite o peso molecular MB: "))
        rho_total = 1.0  # Fixo em 1 como solicitado

        print(f"Parâmetros: rho_a0={rho_a0}, rho_aL={rho_aL}, L={L} (fixo)")
        print(f"           Dab={Dab}, MA={MA}, MB={MB}, rho={rho_total} (fixo)")
    except ValueError:
        print("Erro: Valores inválidos. Usando valores padrão.")
        rho_a0, rho_aL, L = 0.9, 0.1, 1.0
        Dab, MA, MB, rho_total = 1e-5, 28.0, 44.0, 1.0

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

    # DEBUG: Listar arquivos disponíveis
    print(f"\n=== DEBUG: ARQUIVOS DISPONÍVEIS ===")
    available_files = glob.glob(f"{data_path_prefix}_*.xy")
    for file in available_files:
        print(f"Arquivo encontrado: {os.path.basename(file)}")
    print("=== FIM DEBUG ARQUIVOS ===\n")

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

    # Ler arquivo Na_Nb_v.xy para campos vetoriais
    vector_file = f"{data_path_prefix}_Na_Nb_v.xy"
    if os.path.exists(vector_file):
        print(f"Lendo campos vetoriais do arquivo: {os.path.basename(vector_file)}")
        try:
            df_vector = pd.read_csv(vector_file, sep=r"\s+", comment="#", header=None)
            print(f"DEBUG: Arquivo vetorial tem {df_vector.shape[1]} colunas")
            print(f"DEBUG: Primeiras 3 linhas do arquivo vetorial:")
            for i in range(min(3, len(df_vector))):
                print(f"  Linha {i}: {df_vector.iloc[i].tolist()}")

            if df_vector.shape[1] >= 10:
                # Campos vetoriais
                all_data["Na"] = df_vector.iloc[:, 3]  # Coluna 3 = Na_z
                all_data["Nb"] = df_vector.iloc[:, 6]  # Coluna 6 = Nb_z
                all_data["v"] = df_vector.iloc[:, 9]  # Coluna 9 = v_z
                print("Campos vetoriais (componente Z) lidos com sucesso")

                # DEBUG: Mostrar alguns valores para verificar
                print("DEBUG: Valores dos campos vetoriais (primeiros 3 pontos):")
                for i in range(min(3, len(all_data))):
                    print(
                        f"  Ponto {i}: Na={all_data['Na'].iloc[i]:.6f}, Nb={all_data['Nb'].iloc[i]:.6f}, v={all_data['v'].iloc[i]:.6f}"
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
        print("Erro: Arquivo Na_Nb_v.xy não encontrado")

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
        if "Na" in all_data.columns:
            print(f"  Na={all_data['Na'].iloc[i]:.10f}")
        if "Nb" in all_data.columns:
            print(f"  Nb={all_data['Nb'].iloc[i]:.10f}")
        if "v" in all_data.columns:
            print(f"  v={all_data['v'].iloc[i]:.10f}")
    print("=== FIM DEBUG DADOS ===\n")

    # Calcular soluções analíticas COM AS FUNÇÕES VALIDADAS
    print("Calculando soluções analíticas...")
    analytical_solutions = calculate_analytical_solutions(
        all_data["z"].values, rho_a0, rho_aL, L, Dab, MA, MB, rho_total
    )

    # Adicionar soluções analíticas ao DataFrame
    for key, values in analytical_solutions.items():
        all_data[key] = values

    # Calcular erros absolutos
    error_fields = []
    for field in field_names:
        if field in all_data.columns and f"{field}_analytical" in all_data.columns:
            error_field = f"erro_absoluto_{field}"
            all_data[error_field] = all_data[field] - all_data[f"{field}_analytical"]
            error_fields.append(error_field)
            print(f"Erro absoluto calculado para {field}")

    print(f"Dados processados com sucesso: {len(all_data)} pontos")

    # DEBUG: Verificar valores numéricos vs analíticos
    print(f"\n=== DEBUG: COMPARAÇÃO NUMÉRICO vs ANALÍTICO ===")
    for i in range(min(3, len(all_data))):
        print(f"Ponto {i} (z={all_data['z'].iloc[i]:.6f}):")
        for field in field_names:
            if field in all_data.columns and f"{field}_analytical" in all_data.columns:
                num_val = all_data[field].iloc[i]
                ana_val = all_data[f"{field}_analytical"].iloc[i]
                erro = (
                    all_data[f"erro_absoluto_{field}"].iloc[i]
                    if f"erro_absoluto_{field}" in all_data.columns
                    else 0
                )
                print(
                    f"  {field}: num={num_val:.10f}, ana={ana_val:.10f}, erro={erro:.10f}"
                )
    print("=== FIM DEBUG COMPARAÇÃO ===\n")

    # Exportar dados para CSV com 15 casas decimais
    csv_output_path = os.path.join(case_dir, f"{set_name}_equimolar_data.csv")

    # Definir colunas para exportar
    cols_to_export = ["z", "rho", "rho_a", "rho_b"] + field_names

    # Adicionar soluções analíticas
    for field in field_names:
        analytical_field = f"{field}_analytical"
        if analytical_field in all_data.columns:
            cols_to_export.append(analytical_field)

    # Adicionar erros absolutos
    cols_to_export.extend(error_fields)

    # Filtrar apenas colunas que existem
    cols_to_export = [col for col in cols_to_export if col in all_data.columns]

    # Exportar com 15 casas decimais
    all_data[cols_to_export].to_csv(
        csv_output_path, index=False, sep=";", decimal=",", float_format="%.15f"
    )
    print(f"Dados exportados para CSV (15 casas decimais): {csv_output_path}")
    print(f"Colunas exportadas: {cols_to_export}")

    # Plotar gráfico
    plot_from_csv(case_dir, field_names, set_name, rho_a0, rho_aL, L)


def plot_from_csv(
    case_dir, field_names, set_name="myCloud", rho_a0=0.9, rho_aL=0.1, L=1.0
):
    """
    Lê os dados do CSV gerado e plota comparando numérico vs analítico
    """
    print(f"\n--- PLOTANDO DO CSV ---")

    csv_file = os.path.join(case_dir, f"{set_name}_equimolar_data.csv")

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

    # Plotar gráficos para cada campo
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    plot_configs = [
        {
            "field": "rho_a",
            "title": "Concentração Mássica ρ_A",
            "ylabel": "ρ_A [kg/m³]",
        },
        {"field": "Na", "title": "Fluxo Mássico Total N_A", "ylabel": "N_A [kg/m²s]"},
        {"field": "Nb", "title": "Fluxo Mássico Total N_B", "ylabel": "N_B [kg/m²s]"},
        {"field": "v", "title": "Velocidade Média Mássica v", "ylabel": "v [m/s]"},
    ]

    for i, config in enumerate(plot_configs):
        field = config["field"]
        ax = axes[i]

        if field in field_names and field in data.columns:
            # Plotar dados numéricos (OpenFOAM) - TRIÂNGULOS
            ax.plot(
                data["z"],
                data[field],
                "^",
                color="hotpink",
                markersize=8,
                label=f"Numérico: {field}",
                alpha=0.8,
            )

            # Plotar solução analítica - QUADRADOS
            analytical_field = f"{field}_analytical"
            if analytical_field in data.columns:
                ax.plot(
                    data["z"],
                    data[analytical_field],
                    "s",
                    color="purple",
                    markersize=6,
                    label=f"Analítico: {field}",
                    alpha=0.8,
                    fillstyle="none",
                )

        ax.set_xlabel("Posição z [m]")
        ax.set_ylabel(config["ylabel"])
        ax.set_title(config["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L)

    plt.suptitle(
        f"Difusão Equimolar: Comparação Numérico vs Analítico\n(ρ_A0={rho_a0}, ρ_AL={rho_aL})"
    )
    plt.tight_layout()

    plot_output_path = os.path.join(case_dir, f"{set_name}_equimolar_comparison.png")
    plt.savefig(plot_output_path, dpi=300)
    print(f"Gráfico salvo em: {plot_output_path}")
    plt.close()
    print("--- PLOTAGEM CONCLUÍDA ---")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Erro: Especifique pelo menos um campo para processar.")
        print("Uso: python3 postproc.py <campo1> [campo2] ...")
        print("Exemplo: python3 postproc.py rho_a")
        print("Campos disponíveis: rho_a, Na, Nb, v")
        sys.exit(1)

    case_directory = "."
    fields_to_process = sys.argv[1:]

    # Validar campos
    valid_fields = ["rho_a", "Na", "Nb", "v"]
    invalid_fields = [f for f in fields_to_process if f not in valid_fields]
    if invalid_fields:
        print(f"Erro: Campos inválidos: {invalid_fields}")
        print(f"Campos válidos: {valid_fields}")
        sys.exit(1)

    print(f"Processando caso no diretório atual: {os.path.abspath(case_directory)}")
    print(f"Campos: {fields_to_process}")

    run_openfoam_postprocess(case_directory)
    plot_and_export_openfoam_data(case_directory, fields_to_process)
    print("Processo de automação completo concluído.")
