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


def calculate_analytical_solution(z_points, wa0):
    """
    Calcula a solução analítica para rho_a usando a equação:
    rho_a = 1 - 1/(exp(C1*z - C1))
    onde C1 = ln(1 - rho_a0)
    """
    rho_a0 = float(wa0)
    C1 = math.log(1 - rho_a0)
    rho_a_analytical = 1 - 1 / (np.exp(C1 * z_points - C1))
    return rho_a_analytical


def plot_and_export_openfoam_data(
    case_dir, field_names, set_name="myCloud", output_dir="postProcessing/sampleDict"
):
    """
    Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
    """
    print(f"Processando dados para o caso: {case_dir}")

    # Solicitar valor de rho_a0 do usuário
    try:
        rho_a0 = float(
            input("Digite o valor de rho_a0 para cálculo da solução analítica: ")
        )
        print(f"Usando rho_a0 = {rho_a0}")
    except ValueError:
        print("Erro: Valor inválido. Usando rho_a0 = 0.9 como padrão.")
        rho_a0 = 0.9

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

    # Usar o arquivo Na.xy para obter as coordenadas (estrutura mais confiável)
    na_file = f"{data_path_prefix}_Na.xy"
    if os.path.exists(na_file):
        print(f"Lendo coordenadas do arquivo: {os.path.basename(na_file)}")
        try:
            df_na = pd.read_csv(na_file, sep=r"\s+", comment="#", header=None)
            # Estrutura: Z, X, Y, Na
            if df_na.shape[1] >= 3:
                all_data["z"] = df_na.iloc[:, 0]  # Coluna 0 = Z
                all_data["x"] = df_na.iloc[:, 1]  # Coluna 1 = X
                all_data["y"] = df_na.iloc[:, 2]  # Coluna 2 = Y
                print("Coordenadas lidas do arquivo Na.xy")
                if "Na" in field_names:
                    all_data["Na"] = df_na.iloc[:, 3]  # Coluna 3 = Na
                    print("Campo Na lido do arquivo")
        except Exception as e:
            print(f"Erro ao ler coordenadas do arquivo Na: {e}")
            return
    else:
        print("Erro: Arquivo Na.xy não encontrado para ler coordenadas")
        return

    # Ler o arquivo combinado que contém rho, rho_a, rho_b
    combined_file = f"{data_path_prefix}_rho_rho_a_rho_b.xy"
    if os.path.exists(combined_file):
        print(f"Lendo campos do arquivo combinado: {os.path.basename(combined_file)}")
        try:
            df_combined = pd.read_csv(
                combined_file, sep=r"\s+", comment="#", header=None
            )
            print(f"Arquivo combinado tem {df_combined.shape[1]} colunas")

            # ESTRUTURA CORRIGIDA: Já temos coordenadas, só pegar os campos
            # Coluna 0: Z (JÁ TEMOS - IGNORAR)
            # Coluna 1: rho
            # Coluna 2: rho_a
            # Coluna 3: rho_b

            if df_combined.shape[1] >= 4:
                print("Adicionando campos rho, rho_a e rho_b...")
                # IGNORAR coluna 0 (já temos Z das coordenadas)
                if "rho" in field_names:
                    all_data["rho"] = df_combined.iloc[:, 1]  # Coluna 1 = rho
                    print(f"Campo rho lido (coluna 1)")
                if "rho_a" in field_names:
                    all_data["rho_a"] = df_combined.iloc[:, 2]  # Coluna 2 = rho_a
                    print(f"Campo rho_a lido (coluna 2)")
                if "rho_b" in field_names:
                    all_data["rho_b"] = df_combined.iloc[:, 3]  # Coluna 3 = rho_b
                    print(f"Campo rho_b lido (coluna 3)")
            else:
                print(
                    f"Aviso: Arquivo combinado tem apenas {df_combined.shape[1]} colunas"
                )

        except Exception as e:
            print(f"Erro ao ler arquivo combinado: {e}")

    if all_data.empty:
        print("Nenhum dado válido foi lido para plotagem ou exportação.")
        return

    # Calcular solução analítica e erro absoluto
    if "rho_a" in all_data.columns:
        print("Calculando solução analítica e erro absoluto...")
        all_data["rho_a_analytical"] = calculate_analytical_solution(
            all_data["z"], rho_a0
        )
        all_data["erro_absoluto"] = all_data["rho_a"] - all_data["rho_a_analytical"]
        print("Solução analítica e erro absoluto calculados")

    print(f"Dados lidos com sucesso: {len(all_data)} pontos")
    print(f"Colunas disponíveis: {list(all_data.columns)}")

    # DEBUG: Verificar valores
    print("\n=== DEBUG DOS DADOS ===")
    print("Primeiros 5 pontos:")
    for i in range(min(5, len(all_data))):
        print(f"Ponto {i}: Z={all_data['z'].iloc[i]:.15f}")
        print(f"  rho_a numérico = {all_data['rho_a'].iloc[i]:.15f}")
        if "rho_a_analytical" in all_data.columns:
            print(f"  rho_a analítico = {all_data['rho_a_analytical'].iloc[i]:.15f}")
            print(f"  erro absoluto = {all_data['erro_absoluto'].iloc[i]:.15f}")
    print("=== FIM DEBUG ===\n")

    # Exportar dados para CSV com 15 casas decimais
    csv_output_path = os.path.join(case_dir, f"{set_name}_sampled_data.csv")

    # Definir colunas para exportar (incluindo analítico e erro se existirem)
    cols_to_export = [
        col for col in ["x", "y", "z"] if col in all_data.columns
    ] + field_names
    if "rho_a_analytical" in all_data.columns:
        cols_to_export.append("rho_a_analytical")
    if "erro_absoluto" in all_data.columns:
        cols_to_export.append("erro_absoluto")

    # Exportar com 15 casas decimais
    all_data[cols_to_export].to_csv(
        csv_output_path, index=False, sep=";", decimal=",", float_format="%.15f"
    )
    print(f"Dados exportados para CSV (15 casas decimais): {csv_output_path}")

    # Plotar gráfico
    plot_from_csv(case_dir, field_names, set_name, rho_a0)


def plot_from_csv(case_dir, field_names, set_name="myCloud", rho_a0=0.9):
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

    # Verificar se temos a coluna Z e os campos solicitados
    if "z" not in data.columns:
        print("Erro: Coluna 'z' não encontrada no CSV")
        return

    # Plotar gráficos - Apenas perfis de concentração (excluindo 'Na' e 'rho_b')
    fields_to_plot = [f for f in field_names if f not in ["Na", "rho_b"]]

    if not fields_to_plot:
        print("Aviso: Nenhum campo de concentração selecionado para plotagem.")
        return

    plt.figure(figsize=(12, 8))

    # Plotar dados numéricos (OpenFOAM) - TRIÂNGULOS
    for field in fields_to_plot:
        if field in data.columns:
            print(f"Plotando {field}: Z vs {field}")
            plt.plot(
                data["z"],
                data[field],
                "^",  # Triângulo
                color="hotpink",
                markersize=8,
                label=f"Numérico: {field}",
                alpha=0.8,
            )

    # Plotar solução analítica - QUADRADOS (se disponível no CSV ou calcular)
    if "rho_a_analytical" in data.columns:
        plt.plot(
            data["z"],
            data["rho_a_analytical"],
            "s",  # Quadrado
            color="purple",
            markersize=6,
            label=f"Analítico: ρ_a (ρ_a0={rho_a0})",
            alpha=0.8,
            fillstyle="none",
        )
    elif "rho_a" in fields_to_plot:
        # Calcular analítico se não estiver no CSV
        z_points = data["z"]
        rho_a_analytical = calculate_analytical_solution(z_points, rho_a0)
        plt.plot(
            z_points,
            rho_a_analytical,
            "s",  # Quadrado
            color="purple",
            markersize=6,
            label=f"Analítico: ρ_a (ρ_a0={rho_a0})",
            alpha=0.8,
            fillstyle="none",
        )

    plt.xlabel("Posição (z)")
    plt.ylabel("Concentração mássica (ρ)")
    plt.title(f"Comparação: Solução Numérica vs Analítica (ρ_a0={rho_a0})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    plot_output_path = os.path.join(
        case_dir, f"{set_name}_concentration_analytical_plot.png"
    )
    plt.savefig(plot_output_path, dpi=300)
    print(f"Gráfico salvo em: {plot_output_path}")
    plt.close()
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
