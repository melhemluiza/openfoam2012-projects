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


def get_case_type():
    """
    Solicita o tipo de caso ao usuário.
    """
    print("🔍 Selecione o tipo de caso:")
    print("1 - A escoando em B estagnado (stagnantB)")
    print("2 - Difusão equimolar (equimolar)")

    while True:
        choice = input("🎯 Digite 1 ou 2: ").strip()
        if choice == "1":
            return "stagnantB"
        elif choice == "2":
            return "equimolar"
        else:
            print("⚠️  Opção inválida. Digite 1 ou 2.")


def get_user_parameters(case_type):
    """
    Solicita parâmetros do usuário baseado no tipo de caso.
    """
    try:
        if case_type == "stagnantB":
            wa0 = float(
                input("🎯 Digite o valor de wa0 (fração mássica inicial de A): ")
            )
            rho_total = float(
                input("📊 Digite o valor de rho_total (densidade total): ")
            )
            Dab = float(input("🔬 Digite o valor de Dab (coeficiente de difusão): "))
            return {"wa0": wa0, "rho_total": rho_total, "Dab": Dab}

        else:  # equimolar
            wa0 = float(input("🎯 Digite o valor de wa0 (em z=0): "))
            waL = float(input("🎯 Digite o valor de waL (em z=1): "))
            Dab = float(input("🔬 Digite o valor de Dab (coeficiente de difusão): "))
            MA = float(input("⚖️ Digite a massa molar MA (kg/kmol): "))
            MB = float(input("⚖️ Digite a massa molar MB (kg/kmol): "))
            rho_total = float(
                input("📊 Digite a densidade mássica total (rho_total): ")
            )
            return {
                "wa0": wa0,
                "waL": waL,
                "Dab": Dab,
                "MA": MA,
                "MB": MB,
                "rho_total": rho_total,
            }

    except ValueError:
        print("⚠️  Erro: Valores inválidos. Usando valores padrão.")
        if case_type == "stagnantB":
            return {"wa0": 0.9, "rho_total": 1.0, "Dab": 0.1}
        else:
            return {
                "wa0": 0.9,
                "waL": 0.1,
                "Dab": 0.01,
                "MA": 28.96,
                "MB": 44.01,
                "rho_total": 1.0,
            }


def calculate_analytical_solution(z_points, case_type, params, L=1.0):
    """
    Calcula TODAS as variáveis analíticas baseado no tipo de caso.
    """
    if case_type == "stagnantB":
        # SOLUÇÃO PARA A ESCOANDO EM B ESTAGNADO
        wa0 = params["wa0"]
        rho_total = params["rho_total"]
        Dab = params["Dab"]

        rho_total_float = float(rho_total)
        wa0_float = float(wa0)

        rho_a0 = wa0_float * rho_total_float
        rho_b0 = rho_total_float - rho_a0
        K = rho_total_float / rho_b0  # constante

        # Inicializar todos os arrays
        wa_analytical = np.zeros_like(z_points)
        wb_analytical = np.zeros_like(z_points)
        rho_a_analytical = np.zeros_like(z_points)
        rho_b_analytical = np.zeros_like(z_points)

        for i, z in enumerate(z_points):
            if z == 0:
                wa_analytical[i] = wa0_float
                rho_a_analytical[i] = rho_a0
            else:
                rho_a_analytical[i] = rho_total_float - rho_b0 * (K**z)
                wa_analytical[i] = rho_a_analytical[i] / rho_total_float

            rho_b_analytical[i] = rho_total_float - rho_a_analytical[i]
            wb_analytical[i] = rho_b_analytical[i] / rho_total_float

        # Derivadas e fluxos (fora do loop)
        grad_rho_a_analytical = -rho_b0 * (K**z_points) * np.log(K)
        grad_rho_b_analytical = -grad_rho_a_analytical

        ja_analytical = -Dab * grad_rho_a_analytical
        jb_analytical = -Dab * grad_rho_b_analytical

        U_analytical = (1 / (1 - wa_analytical)) * (ja_analytical / rho_total_float)
        Na_analytical = ja_analytical + rho_a_analytical * U_analytical
        Nb_analytical = jb_analytical + rho_b_analytical * U_analytical
        U_ver_analytical = (Na_analytical + Nb_analytical) / rho_total_float

        print(f"wa0_float = {wa0_float}, type = {type(wa0_float)}")
        print(f"wa_analytical[0] = {wa_analytical[0]}, type = {type(wa_analytical[0])}")
        print(f"São iguais? {wa_analytical[0] == wa0_float}")
        print(f"Diferença: {wa_analytical[0] - wa0_float}")

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

    else:
        # SOLUÇÃO PARA DIFUSÃO EQUIMOLAR CORRETA
        wa0 = params["wa0"]
        waL = params["waL"]
        Dab = params["Dab"]
        MA = params["MA"]
        MB = params["MB"]
        rho_total = params["rho_total"]

        rho_a0 = float(wa0) * float(rho_total)
        rho_aL = float(waL) * float(rho_total)
        r = float(MB) / float(MA)

        # Perfil geral (linear)
        rho_a_analytical = rho_a0 + (rho_aL - rho_a0) * (z_points / L)
        rho_b_analytical = rho_total - rho_a_analytical

        # Frações mássicas
        wa_analytical = rho_a_analytical / rho_total
        wb_analytical = rho_b_analytical / rho_total

        # Gradientes (CONSTANTES)
        grad_rho_a = (rho_aL - rho_a0) / L

        # Fluxos difusivos (CONSTANTES no caso equimolar)
        ja_analytical = -Dab * grad_rho_a
        jb_analytical = -ja_analytical  # jb = -ja

        # A velocidade U
        U_analytical = ((1.0 - r) / (1.0 + wa_analytical * (r - 1.0))) * (
            ja_analytical / rho_total
        )

        # Fluxos totais (CONSTANTES no caso equimolar)
        Na_analytical = ja_analytical + rho_a_analytical * U_analytical
        Nb_analytical = jb_analytical + rho_b_analytical * U_analytical

        # Velocidade verificada (deve ser igual a U)
        U_ver_analytical = (Na_analytical + Nb_analytical) / rho_total

        print(f"🔍 DEBUG ANALÍTICO (Equimolar):")
        print(f"   grad_rho_a = {grad_rho_a:.6e}")
        print(f"   ja_analytical = {ja_analytical:.6e} (constante)")
        print(f"   U_analytical médio = {np.mean(U_analytical):.6e} (constante)")
        print(f"   Na_analytical médio = {np.mean(Na_analytical):.6e}")
        print(f"   Nb_analytical médio = {np.mean(Nb_analytical):.6e}")
        print(f"   U_ver_analytical médio = {np.mean(U_ver_analytical):.6e}")

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


def create_plots_directory(case_dir, case_type):
    """
    Cria diretório para os plots se não existir.
    """
    plots_dir = os.path.join(case_dir, f"plots_{case_type}")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"📁 Diretório de plots criado: {plots_dir}")
    return plots_dir


def parse_field_groups(field_args):
    """
    Converte argumentos como 'wawb' em grupos de campos ['wa', 'wb']
    """
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


def plot_field_group(data, field_group, plots_dir, case_type, params, group_name):
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
                # Para campos constantes (caso equimolar), plotar como linha reta
                if (
                    case_type == "equimolar"
                    and len(np.unique(data[analytical_field][valid_mask])) == 1
                ):
                    ax.axhline(
                        y=data[analytical_field].iloc[0],
                        color=color_ana,
                        linestyle="--",
                        linewidth=2,
                        label=f"Analítico: {field}",
                    )
                else:
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

    # Título baseado no tipo de caso
    if case_type == "stagnantB":
        ax.set_title(
            f"Comparação - {group_name} (A escoando em B estagnado)\n"
            f"wa0={params['wa0']}, ρ_total={params['rho_total']}, Dab={params['Dab']}"
        )
    else:
        ax.set_title(
            f"Comparação - {group_name} (Caso Equimolar)\n"
            f"wa0={params['wa0']}, waL={params['waL']}, Dab={params['Dab']}\n"
            f"MA={params['MA']}, MB={params['MB']}, ρ_total={params['rho_total']}"
        )

    plot_filename = f"{group_name}_comparison.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"  💾 Salvo: {plot_filename}")

    plt.close()


def calculate_errors(data, field_groups):
    """
    Calcula erros absolutos e relativos entre soluções numérica e analítica
    e adiciona como colunas no DataFrame.
    """
    print("📊 Calculando erros e adicionando ao DataFrame...")

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
                # Calcular erro absoluto para cada ponto
                data[f"{field}_abs_error"] = np.abs(
                    data[numerical_field] - data[analytical_field]
                )

                # Calcular erro relativo para cada ponto (evitando divisão por zero)
                with np.errstate(divide="ignore", invalid="ignore"):
                    relative_error = np.abs(
                        (data[numerical_field] - data[analytical_field])
                        / (np.abs(data[analytical_field]) + 1e-12)
                    )
                    # Substituir infinitos por NaN
                    relative_error = relative_error.replace([np.inf, -np.inf], np.nan)

                data[f"{field}_rel_error"] = relative_error

                print(f"  ✅ Adicionadas colunas de erro para: {field}")

    return data


def main():
    """
    Função principal para orquestrar o pós-processamento e a plotagem.
    """
    case_dir = os.getcwd()

    # Executar postProcess
    run_openfoam_postprocess(case_dir)

    # Obter tipo de caso e parâmetros
    case_type = get_case_type()
    params = get_user_parameters(case_type)

    print(f"📋 Tipo de caso selecionado: {case_type}")
    print(f"📋 Parâmetros: {params}")

    # Encontrar os arquivos de dados mais recentes
    postprocessing_dir = os.path.join(case_dir, "postProcessing", "sampleDict")

    # Buscar as pastas de tempo
    time_dirs = glob.glob(os.path.join(postprocessing_dir, "*"))
    if not time_dirs:
        print(f"❌ Erro: Nenhuma pasta de tempo encontrada em {postprocessing_dir}")
        sys.exit(1)

    latest_time_dir = max(time_dirs, key=os.path.getctime)
    print(f"📂 Diretório de tempo mais recente: {os.path.basename(latest_time_dir)}")

    # Arquivos específicos
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

    # Atribuir nomes corretos às colunas
    if df_cloud1.shape[1] == 6:
        df_cloud1.columns = ["z", "rho", "rho_a", "rho_b", "wa", "wb"]
        print("✅ Colunas do arquivo 1 identificadas: z, rho, rho_a, rho_b, wa, wb")
    else:
        print(f"⚠️  Arquivo 1 tem {df_cloud1.shape[1]} colunas, usando nomes genéricos")
        df_cloud1.columns = [f"col_{i}" for i in range(df_cloud1.shape[1])]
        if df_cloud1.shape[1] >= 1:
            df_cloud1 = df_cloud1.rename(columns={"col_0": "z"})

    # Ler o segundo arquivo (Na, Nb, U, U_ver, ja, jb) - 19 colunas
    df_cloud2 = read_openfoam_data(cloud2_file)
    if df_cloud2 is None:
        sys.exit(1)

    # Estrutura para 19 colunas
    if df_cloud2.shape[1] == 19:
        df_cloud2.columns = [
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
            "jb_z",
        ]
        print("✅ Colunas do arquivo 2 identificadas (19 colunas)")
    else:
        print(f"⚠️  Arquivo 2 tem {df_cloud2.shape[1]} colunas, usando nomes genéricos")
        df_cloud2.columns = [f"col_{i}" for i in range(df_cloud2.shape[1])]
        if df_cloud2.shape[1] >= 1:
            df_cloud2 = df_cloud2.rename(columns={"col_0": "z"})

    # Combinar os dois DataFrames baseado na coluna z
    data_combined = pd.merge(df_cloud1, df_cloud2, on="z", how="inner")
    print(f"📊 Dados combinados: {data_combined.shape[0]} pontos comuns")

    # Usar apenas componentes Z para domínio 1D
    if "Na_z" in data_combined.columns:
        data_combined["Na"] = data_combined["Na_z"]
    if "Nb_z" in data_combined.columns:
        data_combined["Nb"] = data_combined["Nb_z"]
    if "U_z" in data_combined.columns:
        data_combined["U"] = data_combined["U_z"]
    if "U_ver_z" in data_combined.columns:
        data_combined["U_ver"] = data_combined["U_ver_z"]
    if "ja_z" in data_combined.columns:
        data_combined["ja"] = data_combined["ja_z"]
    if "jb_z" in data_combined.columns:
        data_combined["jb"] = data_combined["jb_z"]

    # Calcular U_ver (verificação: (Na + Nb) / rho) se não existir
    if (
        "U_ver" not in data_combined.columns
        and "Na" in data_combined.columns
        and "Nb" in data_combined.columns
        and "rho" in data_combined.columns
    ):
        data_combined["U_ver"] = (
            data_combined["Na"] + data_combined["Nb"]
        ) / data_combined["rho"]

    # Calcular fluxos difusivos (ja e jb) se não existirem
    if (
        "ja" not in data_combined.columns
        and "rho_a" in data_combined.columns
        and "U" in data_combined.columns
        and "Na" in data_combined.columns
    ):
        data_combined["ja"] = (
            data_combined["Na"] - data_combined["rho_a"] * data_combined["U"]
        )

    if (
        "jb" not in data_combined.columns
        and "rho_b" in data_combined.columns
        and "U" in data_combined.columns
        and "Nb" in data_combined.columns
    ):
        data_combined["jb"] = (
            data_combined["Nb"] - data_combined["rho_b"] * data_combined["U"]
        )

    # Calcular a solução analítica baseada no tipo de caso
    z_points = data_combined["z"]
    analytical_solutions = calculate_analytical_solution(z_points, case_type, params)

    # Adicionar soluções analíticas ao DataFrame
    for col, values in analytical_solutions.items():
        data_combined[f"{col}_analytical"] = values

    # Configurar pandas para mostrar mais casas decimais
    pd.set_option("display.float_format", "{:.16e}".format)
    np.set_printoptions(precision=16)

    # Obter grupos de campos dos argumentos da linha de comando
    if len(sys.argv) < 2:
        print("Uso: python3 postproc.py <grupo1> <grupo2> ...")
        print("Exemplo: python3 postproc.py wawb NaNb jajb UU_ver")
        print("Campos disponíveis: rho_a, rho_b, wa, wb, U, ja, jb, Na, Nb, U_ver")
        sys.exit(1)

    field_args = sys.argv[1:]
    field_groups = parse_field_groups(field_args)

    print(f"📋 Grupos de campos a plotar: {field_groups}")

    # Calcular erros e adicionar ao DataFrame
    data_combined = calculate_errors(data_combined, field_groups)

    # Salvar o DataFrame combinado com erros em um arquivo CSV
    combined_csv_path = os.path.join(
        case_dir, f"combined_data_{case_type}_with_errors.csv"
    )
    data_combined.to_csv(combined_csv_path, index=False, float_format="%.16e")
    print(f"💾 DataFrame combinado com erros salvo em: {combined_csv_path}")
    print(f"📋 Colunas disponíveis: {list(data_combined.columns)}")

    # Mostrar primeiras linhas com alta precisão
    print("📊 Primeiras 3 linhas dos dados combinados (com erros):")
    print(data_combined.head(3).to_string(float_format="%.16e"))

    # Criar diretório de plots
    plots_dir = create_plots_directory(case_dir, case_type)

    # Plotar cada grupo de campos
    for i, field_group in enumerate(field_groups):
        group_name = "_".join(field_group)
        plot_field_group(
            data_combined,
            field_group,
            plots_dir,
            case_type,
            params,
            group_name,
        )

    print(f"🎉 Processo concluído para caso {case_type}!")


if __name__ == "__main__":
    main()
