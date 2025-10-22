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
    print("🚀 Iniciando o pós-processamento do OpenFOAM...")
    try:
        result = subprocess.run(
            ["postProcess", "-func", "sampleDict", "-latestTime"],
            cwd=case_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Pós-processamento do OpenFOAM concluído com sucesso!")
        if result.stderr:
            print("📝 Avisos do postProcess:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante o pós-processamento do OpenFOAM: {e}")
        print(f"🔍 Saída de erro: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(
            "❌ Erro: Comando 'postProcess' não encontrado. Verifique o ambiente OpenFOAM."
        )
        sys.exit(1)


def calculate_analytical_solution(z_points, wa0, rho_total, Dab):
    """
    Calcula TODAS as variáveis analíticas a partir da solução de rho_a.
    """
    rho_a0 = float(wa0) * float(rho_total)

    rho_a_analytical = rho_total - ((rho_total - rho_a0) / ((1 - wa0) ** z_points))
    rho_b_analytical = rho_total - rho_a_analytical
    wa_analytical = rho_a_analytical / rho_total
    wb_analytical = rho_b_analytical / rho_total

    grad_rho_a_analytical = ((rho_total - rho_a0) * np.log(1 - wa0)) / (
        (1 - wa0) ** z_points
    )
    grad_rho_b_analytical = -grad_rho_a_analytical

    ja_analytical = -Dab * grad_rho_a_analytical
    jb_analytical = -Dab * grad_rho_b_analytical

    U_analytical = (1 / (1 - wa_analytical)) * (ja_analytical / rho_total)
    Na_analytical = ja_analytical + rho_a_analytical * U_analytical
    Nb_analytical = jb_analytical + rho_b_analytical * U_analytical
    U_ver_analytical = (Na_analytical + Nb_analytical) / rho_total

    return {
        "rho_a": rho_a_analytical,
        "rho_b": rho_b_analytical,
        "wa": wa_analytical,
        "wb": wb_analytical,
        "ja": ja_analytical,
        "jb": jb_analytical,
        "U": U_analytical,
        "Na": Na_analytical,
        "Nb": Nb_analytical,
        "U_ver": U_ver_analytical,
    }


def get_user_parameters():
    """
    Solicita parâmetros do usuário para cálculo analítico.
    """
    try:
        wa0 = float(input("🎯 Digite o valor de wa0 (fração mássica inicial de A): "))
        rho_total = float(input("📊 Digite o valor de rho_total (densidade total): "))
        Dab = float(input("🔬 Digite o valor de Dab (coeficiente de difusão): "))
        return wa0, rho_total, Dab
    except ValueError:
        print(
            "⚠️  Erro: Valores inválidos. Usando valores padrão: wa0=0.9, rho_total=1.0, Dab=0.1"
        )
        return 0.9, 1.0, 0.1


def read_openfoam_data(file_path):
    """
    Lê arquivos de dados do OpenFOAM e retorna DataFrame.
    """
    try:
        # Lê o arquivo OpenFOAM (formato com espaços)
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
        print(
            f"📄 Arquivo {os.path.basename(file_path)}: {df.shape[1]} colunas, {df.shape[0]} pontos"
        )
        return df
    except Exception as e:
        print(f"❌ Erro ao ler arquivo {file_path}: {e}")
        return None


def create_plots_directory(case_dir):
    """
    Cria diretório para os plots se não existir.
    """
    plots_dir = os.path.join(case_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"📁 Diretório de plots criado: {plots_dir}")
    return plots_dir


def parse_field_groups(field_args):
    """
    Converte argumentos como 'wawb' em grupos de campos ['wa', 'wb']
    """
    field_mapping = {
        "wa": "wa",
        "wb": "wb",
        "rho_a": "rho_a",
        "rho_b": "rho_b",
        "ja": "ja",
        "jb": "jb",
        "U": "U",
        "Na": "Na",
        "Nb": "Nb",
        "U_ver": "U_ver",
    }

    field_groups = []

    for arg in field_args:
        # Verifica se é um grupo composto (ex: 'wawb')
        individual_fields = []
        temp_arg = arg

        # Tenta encontrar campos individuais no argumento composto
        for field_name in [
            "U_ver",
            "rho_a",
            "rho_b",
            "wa",
            "wb",
            "ja",
            "jb",
            "U",
            "Na",
            "Nb",
        ]:
            if field_name in temp_arg:
                individual_fields.append(field_name)
                temp_arg = temp_arg.replace(field_name, "")

        # Se encontrou campos individuais, usa eles
        if individual_fields:
            field_groups.append(individual_fields)
        else:
            # Se não, trata como campo individual
            field_groups.append([arg])

    return field_groups


def plot_field_group(data, field_group, plots_dir, wa0, rho_total, Dab, group_name):
    """
    Plota um grupo de campos na mesma imagem.
    """
    print(f"🎨 Plotando grupo {group_name}: {field_group}")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(field_group) * 2))
    markers_num = ["^", "s", "o", "d", "v", "<", ">", "p"]
    markers_ana = ["s", "d", "o", "^", ">", "v", "<", "p"]

    has_data = False

    for idx, field in enumerate(field_group):
        numerical_field = field
        analytical_field = f"{field}_analytical"

        color_num = colors[idx * 2]
        color_ana = colors[idx * 2 + 1]
        marker_num = markers_num[idx % len(markers_num)]
        marker_ana = markers_ana[idx % len(markers_ana)]

        # Plotar dados numéricos se disponíveis
        if numerical_field in data.columns:
            valid_mask = ~(
                data[numerical_field].isna() | np.isinf(data[numerical_field])
            )
            if valid_mask.any():
                ax.plot(
                    data["z"][valid_mask],
                    data[numerical_field][valid_mask],
                    marker_num,
                    color=color_num,
                    markersize=8,
                    label=f"Numérico: {field}",
                    alpha=0.8,
                    linewidth=2,
                )
                has_data = True
                print(f"  ✅ Plotando numérico: {field}")

        # Plotar dados analíticos se disponíveis
        if analytical_field in data.columns:
            valid_mask = ~(
                data[analytical_field].isna() | np.isinf(data[analytical_field])
            )
            if valid_mask.any():
                ax.plot(
                    data["z"][valid_mask],
                    data[analytical_field][valid_mask],
                    marker_ana,
                    color=color_ana,
                    markersize=6,
                    label=f"Analítico: {field}",
                    alpha=0.8,
                    linestyle="--",
                    linewidth=2,
                )
                has_data = True
                print(f"  ✅ Plotando analítico: {field}")

    if not has_data:
        print(f"  ⚠️  Nenhum dado válido encontrado para o grupo {field_group}")
        plt.close()
        return

    ax.set_xlabel("Posição (z)")
    ax.set_ylabel("Valores")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Comparação - {group_name} (wa0={wa0}, ρ_total={rho_total}, Dab={Dab})"
    )

    plot_filename = f"{group_name}_comparison.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"  💾 Salvo: {plot_filename}")

    plt.close()


def calculate_errors(data, field_groups):
    """
    Calcula erros absolutos e relativos entre soluções numérica e analítica.
    """
    print("📊 Calculando erros...")

    errors = {}

    # Extrair todos os campos individuais dos grupos
    all_fields = []
    for group in field_groups:
        all_fields.extend(group)

    for field in set(all_fields):  # Remove duplicatas
        numerical_field = field
        analytical_field = f"{field}_analytical"

        if numerical_field in data.columns and analytical_field in data.columns:
            # Remover NaN e infinitos
            mask = ~(
                data[numerical_field].isna()
                | data[analytical_field].isna()
                | np.isinf(data[numerical_field])
                | np.isinf(data[analytical_field])
            )

            if mask.any():
                num_values = data[numerical_field][mask]
                ana_values = data[analytical_field][mask]

                # Erro absoluto
                absolute_error = np.abs(num_values - ana_values)
                max_abs_error = np.max(absolute_error)
                mean_abs_error = np.mean(absolute_error)

                # Erro relativo (evitando divisão por zero)
                relative_error = np.abs(
                    (num_values - ana_values) / (ana_values + 1e-12)
                )
                max_rel_error = np.max(relative_error)
                mean_rel_error = np.mean(relative_error)

                errors[field] = {
                    "max_absolute_error": max_abs_error,
                    "mean_absolute_error": mean_abs_error,
                    "max_relative_error": max_rel_error,
                    "mean_relative_error": mean_rel_error,
                }

                print(f"  📈 {field}:")
                print(f"     Erro Absoluto Máximo: {max_abs_error:.6e}")
                print(f"     Erro Absoluto Médio: {mean_abs_error:.6e}")
                print(f"     Erro Relativo Máximo: {max_rel_error:.6e}")
                print(f"     Erro Relativo Médio: {mean_rel_error:.6e}")

    return errors


def main():
    """
    Função principal para orquestrar o pós-processamento e a plotagem.
    """
    case_dir = os.getcwd()

    # Executar postProcess
    run_openfoam_postprocess(case_dir)

    # Obter parâmetros do usuário
    wa0, rho_total, Dab = get_user_parameters()

    # Encontrar os arquivos de dados mais recentes
    postprocessing_dir = os.path.join(case_dir, "postProcessing", "sampleDict")

    # Buscar as pastas de tempo
    time_dirs = glob.glob(os.path.join(postprocessing_dir, "*"))
    if not time_dirs:
        print(f"❌ Erro: Nenhuma pasta de tempo encontrada em {postprocessing_dir}")
        sys.exit(1)

    latest_time_dir = max(time_dirs, key=os.path.getctime)
    print(f"📂 Diretório de tempo mais recente: {os.path.basename(latest_time_dir)}")

    # Arquivos específicos que existem
    cloud1_file = os.path.join(latest_time_dir, "myCloud_rho_rho_a_rho_b_wa_wb.xy")
    cloud2_file = os.path.join(latest_time_dir, "myCloud_Na_Nb_U_U_ver_ja_jb.xy")

    if not os.path.exists(cloud1_file):
        print(f"❌ Erro: Arquivo {cloud1_file} não encontrado")
        sys.exit(1)
    if not os.path.exists(cloud2_file):
        print(f"❌ Erro: Arquivo {cloud2_file} não encontrado")
        sys.exit(1)

    print(f"📂 Arquivo 1 encontrado: {os.path.basename(cloud1_file)}")
    print(f"📂 Arquivo 2 encontrado: {os.path.basename(cloud2_file)}")

    # Ler o primeiro arquivo (rho, rho_a, rho_b, wa, wb)
    df_cloud1 = read_openfoam_data(cloud1_file)
    if df_cloud1 is None:
        sys.exit(1)

    # Atribuir nomes às colunas baseado na descrição
    if df_cloud1.shape[1] >= 6:  # z, rho, rho_a, rho_b, wa, wb
        df_cloud1.columns = ["z", "rho", "rho_a", "rho_b", "wa", "wb"][
            : df_cloud1.shape[1]
        ]
    else:
        print(
            f"⚠️  Arquivo 1 tem apenas {df_cloud1.shape[1]} colunas, usando nomes genéricos"
        )
        df_cloud1.columns = [f"col_{i}" for i in range(df_cloud1.shape[1])]
        # Assumindo que a primeira coluna é sempre z
        if df_cloud1.shape[1] >= 1:
            df_cloud1 = df_cloud1.rename(columns={"col_0": "z"})

    # Ler o segundo arquivo (Na, Nb, U, U_ver, ja, jb) - CORRIGIDO: 19 colunas
    df_cloud2 = read_openfoam_data(cloud2_file)
    if df_cloud2 is None:
        sys.exit(1)

    # CORREÇÃO: 19 colunas ao invés de 21 (jb só tem componente z)
    expected_columns_cloud2 = [
        "z",
        "x",
        "y",
        "Na_z",
        "Na_x",
        "Na_y",
        "Nb_z",
        "Nb_x",
        "Nb_y",
        "U_z",
        "U_x",
        "U_y",
        "U_ver_z",
        "U_ver_x",
        "U_ver_y",
        "ja_z",
        "ja_x",
        "ja_y",
        "jb_z",  # Apenas jb_z, sem jb_x e jb_y
    ]

    if df_cloud2.shape[1] == len(expected_columns_cloud2):
        df_cloud2.columns = expected_columns_cloud2
    else:
        print(
            f"⚠️  Arquivo 2 tem {df_cloud2.shape[1]} colunas (esperadas {len(expected_columns_cloud2)}), usando nomes genéricos"
        )
        df_cloud2.columns = [f"col_{i}" for i in range(df_cloud2.shape[1])]
        # Assumindo que a primeira coluna é sempre z
        if df_cloud2.shape[1] >= 1:
            df_cloud2 = df_cloud2.rename(columns={"col_0": "z"})

    # Combinar os dois DataFrames baseado na coluna z
    data_combined = pd.merge(df_cloud1, df_cloud2, on="z", how="inner")
    print(f"📊 Dados combinados: {data_combined.shape[0]} pontos comuns")

    # Calcular magnitudes para campos vetoriais - CORRIGIDO para jb
    if all(col in data_combined.columns for col in ["U_x", "U_y", "U_z"]):
        data_combined["U"] = np.sqrt(
            data_combined["U_x"] ** 2
            + data_combined["U_y"] ** 2
            + data_combined["U_z"] ** 2
        )

    if all(col in data_combined.columns for col in ["ja_x", "ja_y", "ja_z"]):
        data_combined["ja"] = np.sqrt(
            data_combined["ja_x"] ** 2
            + data_combined["ja_y"] ** 2
            + data_combined["ja_z"] ** 2
        )

    # CORREÇÃO: jb só tem componente z
    if "jb_z" in data_combined.columns:
        data_combined["jb"] = np.abs(
            data_combined["jb_z"]
        )  # Usar valor absoluto já que é 1D

    if all(col in data_combined.columns for col in ["Na_x", "Na_y", "Na_z"]):
        data_combined["Na"] = np.sqrt(
            data_combined["Na_x"] ** 2
            + data_combined["Na_y"] ** 2
            + data_combined["Na_z"] ** 2
        )

    if all(col in data_combined.columns for col in ["Nb_x", "Nb_y", "Nb_z"]):
        data_combined["Nb"] = np.sqrt(
            data_combined["Nb_x"] ** 2
            + data_combined["Nb_y"] ** 2
            + data_combined["Nb_z"] ** 2
        )

    if all(col in data_combined.columns for col in ["U_ver_x", "U_ver_y", "U_ver_z"]):
        data_combined["U_ver"] = np.sqrt(
            data_combined["U_ver_x"] ** 2
            + data_combined["U_ver_y"] ** 2
            + data_combined["U_ver_z"] ** 2
        )

    # Calcular a solução analítica
    z_points = data_combined["z"]
    analytical_solutions = calculate_analytical_solution(z_points, wa0, rho_total, Dab)

    # Adicionar soluções analíticas ao DataFrame
    for col, values in analytical_solutions.items():
        data_combined[f"{col}_analytical"] = values

    # Configurar pandas para mostrar mais casas decimais
    pd.set_option("display.float_format", "{:.16e}".format)
    np.set_printoptions(precision=16)

    # Salvar o DataFrame combinado em um arquivo CSV para depuração
    combined_csv_path = os.path.join(case_dir, "combined_data.csv")
    data_combined.to_csv(combined_csv_path, index=False, float_format="%.16e")
    print(f"💾 DataFrame combinado salvo em: {combined_csv_path}")
    print(f"📋 Colunas disponíveis: {list(data_combined.columns)}")

    # Mostrar primeiras linhas com alta precisão
    print("📊 Primeiras 3 linhas dos dados combinados:")
    print(data_combined.head(3).to_string(float_format="%.16e"))

    # Criar diretório de plots
    plots_dir = create_plots_directory(case_dir)

    # Obter grupos de campos dos argumentos da linha de comando
    if len(sys.argv) < 2:
        print("Uso: python3 postproc.py <grupo1> <grupo2> ...")
        print("Exemplo: python3 postproc.py wawb NaNb jajb UU_ver")
        print("Campos disponíveis: rho_a, rho_b, wa, wb, U, ja, jb, Na, Nb, U_ver")
        sys.exit(1)

    field_args = sys.argv[1:]
    field_groups = parse_field_groups(field_args)

    print(f"📋 Grupos de campos a plotar: {field_groups}")

    # Calcular erros
    errors = calculate_errors(data_combined, field_groups)

    # Salvar erros em arquivo
    errors_csv_path = os.path.join(plots_dir, "errors_analysis.csv")
    errors_df = pd.DataFrame(errors).T
    errors_df.to_csv(errors_csv_path, float_format="%.16e")
    print(f"💾 Análise de erros salva em: {errors_csv_path}")

    # Plotar cada grupo de campos
    for i, field_group in enumerate(field_groups):
        group_name = "_".join(field_group)
        plot_field_group(
            data_combined, field_group, plots_dir, wa0, rho_total, Dab, group_name
        )

    print("🎉 Processo concluído!")


if __name__ == "__main__":
    main()
