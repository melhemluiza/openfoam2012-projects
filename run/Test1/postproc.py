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

    Equação analítica para rho_a:
    rho_a = 1 - 1/(exp(C1*z - C1))
    onde C1 = ln(1 - rho_a0)

    A partir de rho_a, calculamos todas as outras variáveis.
    """
    rho_a0 = float(wa0) * float(rho_total)  # Converter wa0 para rho_a0

    # CORREÇÃO: Verificar se rho_a0 é válido para evitar NaN
    if rho_a0 >= 1.0:
        print(f"⚠️  Aviso: rho_a0 = {rho_a0} é >= 1.0, ajustando para 0.999")
        rho_a0 = 0.999 * float(rho_total)  # Ajuste para evitar problemas numéricos

    C1 = np.log(1 - rho_a0)

    # Calcular rho_a analítico
    rho_a_analytical = 1 - 1 / (np.exp(C1 * z_points - C1))

    # Calcular todas as outras variáveis analíticas
    rho_b_analytical = rho_total - rho_a_analytical
    wa_analytical = rho_a_analytical / rho_total
    wb_analytical = rho_b_analytical / rho_total

    # Calcular gradientes analíticos (derivada de rho_a em relação a z)
    # d(rho_a)/dz = C1 / exp(C1 * z - C1)
    grad_rho_a_analytical = C1 / np.exp(C1 * z_points - C1)
    grad_rho_b_analytical = -grad_rho_a_analytical

    # Calcular fluxos difusivos
    ja_analytical = -Dab * grad_rho_a_analytical
    jb_analytical = -Dab * grad_rho_b_analytical

    # Calcular velocidade e fluxos totais
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


def read_combined_file(file_path, expected_columns):
    """
    Lê arquivo combinado e retorna DataFrame com as colunas especificadas.
    """
    try:
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None)
        print(
            f"📄 Arquivo {os.path.basename(file_path)}: {df.shape[1]} colunas, {df.shape[0]} pontos"
        )

        # Verificar se temos colunas suficientes
        if df.shape[1] < len(expected_columns):
            print(
                f"⚠️  Aviso: Arquivo tem apenas {df.shape[1]} colunas, esperava {len(expected_columns)}"
            )
            # Retornar apenas as colunas disponíveis
            available_columns = expected_columns[: df.shape[1]]
            result_df = pd.DataFrame()
            for i, col_name in enumerate(available_columns):
                result_df[col_name] = df.iloc[:, i]
            return result_df
        else:
            # Criar DataFrame com todas as colunas esperadas
            result_df = pd.DataFrame()
            for i, col_name in enumerate(expected_columns):
                result_df[col_name] = df.iloc[:, i]
            return result_df
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


def plot_individual_fields(data, field_names, plots_dir, wa0, rho_total, Dab):
    """
    Plota cada campo em uma imagem separada.
    """
    print(f"🎨 Plotando {len(field_names)} campos individuais...")

    for field in field_names:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Verificar se temos dados numéricos e analíticos para este campo
        numerical_field = field
        analytical_field = f"{field}_analytical"

        # Para campos vetoriais, usar componente x (mas sem _x no nome analítico)
        if field in ["U", "U_ver", "ja", "jb", "Na", "Nb"]:
            numerical_field = f"{field}_x"  # Dados numéricos têm _x
            # analytical_field já está correto (sem _x)

        print(f"  🔍 Verificando campo {field}:")
        print(f"     Numérico: {numerical_field} -> {numerical_field in data.columns}")
        print(
            f"     Analítico: {analytical_field} -> {analytical_field in data.columns}"
        )

        if numerical_field in data.columns:
            print(f"     Valores numéricos: {data[numerical_field].head(3).tolist()}")
        if analytical_field in data.columns:
            print(f"     Valores analíticos: {data[analytical_field].head(3).tolist()}")

        if numerical_field in data.columns and analytical_field in data.columns:
            # Plotar numérico
            ax.plot(
                data["z"],
                data[numerical_field],
                "^",
                color="hotpink",
                markersize=6,
                label=f"Numérico: {field}",
                alpha=0.8,
            )

            # Plotar analítico
            ax.plot(
                data["z"],
                data[analytical_field],
                "s",
                color="purple",
                markersize=4,
                label=f"Analítico: {field}",
                alpha=0.8,
                fillstyle="none",
            )

            ax.set_xlabel("Posição (z)")
            ax.set_ylabel(field)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_title(f"{field} - wa0={wa0}, ρ_total={rho_total}, Dab={Dab}")

            # Salvar imagem individual
            plot_filename = f"{field}_comparison.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"  💾 Salvo: {plot_filename}")
        else:
            print(f"  ⚠️  Dados insuficientes para {field}")

        plt.close()


def plot_combined_fields(data, combined_fields, plots_dir, wa0, rho_total, Dab):
    """
    Plota campos combinados na mesma imagem - TODOS JUNTOS NO MESMO GRÁFICO.
    """
    print(f"🎨 Plotando campos combinados no MESMO GRÁFICO: {combined_fields}")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    markers_num = ["^", "s", "o", "d", "v", "<", ">", "p"]
    markers_ana = ["s", "d", "o", "^", ">", "v", "<", "p"]

    plotted_something = False

    for idx, field in enumerate(combined_fields):
        # Verificar se temos dados numéricos e analíticos para este campo
        numerical_field = field
        analytical_field = f"{field}_analytical"

        # Para campos vetoriais, usar componente x (mas sem _x no nome analítico)
        if field in ["U", "U_ver", "ja", "jb", "Na", "Nb"]:
            numerical_field = f"{field}_x"  # Dados numéricos têm _x
            # analytical_field já está correto (sem _x)

        color_num = colors[idx * 2 % len(colors)]
        color_ana = colors[(idx * 2 + 1) % len(colors)]
        marker_num = markers_num[idx % len(markers_num)]
        marker_ana = markers_ana[idx % len(markers_ana)]

        print(f"  🔍 Processando {field}:")
        print(f"     Numérico: {numerical_field} -> {numerical_field in data.columns}")
        print(
            f"     Analítico: {analytical_field} -> {analytical_field in data.columns}"
        )

        if numerical_field in data.columns:
            print(
                f"     Valores numéricos (primeiros 3): {data[numerical_field].head(3).tolist()}"
            )
            # Plotar numérico
            ax.plot(
                data["z"],
                data[numerical_field],
                marker_num,
                color=color_num,
                markersize=6,
                label=f"Numérico {field}",
                alpha=0.8,
                linewidth=2,
            )
            plotted_something = True

        if analytical_field in data.columns:
            print(
                f"     Valores analíticos (primeiros 3): {data[analytical_field].head(3).tolist()}"
            )
            # Plotar analítico
            ax.plot(
                data["z"],
                data[analytical_field],
                marker_ana,
                color=color_ana,
                markersize=4,
                label=f"Analítico {field}",
                alpha=0.8,
                fillstyle="none",
                linestyle="--",
                linewidth=2,
            )
            plotted_something = True

    if not plotted_something:
        print(f"  ⚠️  Nenhum dado encontrado para os campos {combined_fields}")
        plt.close()
        return

    ax.set_xlabel("Posição (z)", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Criar nome do arquivo combinado
    combined_name = "".join(combined_fields)
    plot_title = " vs ".join(combined_fields)
    ax.set_title(
        f"{plot_title} - wa0={wa0}, ρ_total={rho_total}, Dab={Dab}", fontsize=14
    )

    # Salvar imagem combinada
    plot_filename = f"{combined_name}_combined.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"  💾 Salvo: {plot_filename} (todos os campos no mesmo gráfico)")
    plt.close()


def plot_and_export_openfoam_data(
    case_dir, field_names, set_name="myCloud", output_dir="postProcessing/sampleDict"
):
    """
    Lê os dados de amostragem do OpenFOAM, plota gráficos e exporta para CSV.
    Calcula soluções analíticas para todos os campos.
    """
    print(f"🔍 Processando dados para o caso: {case_dir}")

    # Solicitar parâmetros do usuário
    wa0, rho_total, Dab = get_user_parameters()
    print(f"📋 Parâmetros: wa0={wa0}, rho_total={rho_total}, Dab={Dab}")

    # Criar diretório para plots
    plots_dir = create_plots_directory(case_dir)

    post_processing_path = os.path.join(case_dir, output_dir)
    if not os.path.exists(post_processing_path):
        print(
            f"❌ Erro: Diretório de pós-processamento não encontrado em {post_processing_path}"
        )
        return

    time_dirs = [
        d
        for d in os.listdir(post_processing_path)
        if os.path.isdir(os.path.join(post_processing_path, d))
    ]
    if not time_dirs:
        print(
            f"❌ Erro: Nenhum diretório de tempo encontrado em {post_processing_path}"
        )
        return

    latest_time = sorted(time_dirs, key=float)[-1]
    data_path = os.path.join(post_processing_path, latest_time)

    print(f"📁 Procurando arquivos em: {data_path}")

    # DEBUG: Listar o que existe no diretório
    if os.path.exists(data_path):
        print(f"📂 Conteúdo do diretório {data_path}:")
        for item in os.listdir(data_path):
            print(f"   📝 {item}")
    else:
        print(f"❌ Diretório não existe: {data_path}")
        return

    all_data = pd.DataFrame()

    # DEFINIR OS ARQUIVOS COMBINADOS COM COLUNAS CORRETAS
    scalar_files = {
        "myCloud_rho_rho_a_rho_b_wa_wb": ["z", "rho", "rho_a", "rho_b", "wa", "wb"],
    }

    # ATUALIZADO: Ajustar colunas esperadas para o arquivo vetorial baseado na realidade (19 colunas)
    vector_files = {
        "myCloud_Na_Nb_U_U_ver_ja_jb": [
            "z",  # 0
            "x",  # 1
            "y",  # 2
            "Na_x",  # 3
            "Na_y",  # 4
            "Na_z",  # 5
            "Nb_x",  # 6
            "Nb_y",  # 7
            "Nb_z",  # 8
            "U_x",  # 9
            "U_y",  # 10
            "U_z",  # 11
            "U_ver_x",  # 12
            "U_ver_y",  # 13
            "U_ver_z",  # 14
            "ja_x",  # 15
            "ja_y",  # 16
            "ja_z",  # 17
            "jb_x",  # 18
        ],
    }

    # PRIMEIRO: Encontrar e ler arquivo de coordenadas
    coord_file = None

    # Estratégia: Procurar por arquivos específicos que sabemos que existem
    all_possible_patterns = list(scalar_files.keys()) + list(vector_files.keys())

    for pattern in all_possible_patterns:
        test_file = os.path.join(data_path, f"{pattern}.xy")
        if os.path.exists(test_file):
            coord_file = test_file
            print(f"✅ Arquivo encontrado: {os.path.basename(test_file)}")
            break

    if coord_file is None:
        print(f"❌ Erro: Nenhum arquivo .xy encontrado em {data_path}")
        print("📂 Arquivos disponíveis:")
        for file in os.listdir(data_path):
            print(f"   📝 {file}")
        return

    print(f"📖 Lendo coordenadas do arquivo: {os.path.basename(coord_file)}")

    # SEGUNDO: Ler todos os arquivos combinados disponíveis
    print("\n🔍 Lendo dados dos arquivos combinados...")

    # Ler arquivo de ESCALARES
    scalar_file = os.path.join(data_path, "myCloud_rho_rho_a_rho_b_wa_wb.xy")
    if os.path.exists(scalar_file):
        print(f"📊 Lendo arquivo de escalares: {os.path.basename(scalar_file)}")
        df_scalar = read_combined_file(
            scalar_file, ["z", "rho", "rho_a", "rho_b", "wa", "wb"]
        )
        if df_scalar is not None:
            # Adicionar colunas que ainda não existem em all_data
            for col in df_scalar.columns:
                all_data[col] = df_scalar[col]
                print(f"  ➕ Adicionado campo escalar: {col}")
    else:
        print("⚠️  Arquivo de escalares não encontrado")

    # Ler arquivo de VETORES
    vector_file = os.path.join(data_path, "myCloud_Na_Nb_U_U_ver_ja_jb.xy")
    if os.path.exists(vector_file):
        print(f"📈 Lendo arquivo de vetores: {os.path.basename(vector_file)}")
        df_vector = read_combined_file(
            vector_file,
            [
                "z",
                "x",
                "y",
                "Na_x",
                "Na_y",
                "Na_z",
                "Nb_x",
                "Nb_y",
                "Nb_z",
                "U_x",
                "U_y",
                "U_z",
                "U_ver_x",
                "U_ver_y",
                "U_ver_z",
                "ja_x",
                "ja_y",
                "ja_z",
                "jb_x",  # Apenas 19 colunas disponíveis
            ],
        )
        if df_vector is not None:
            # Adicionar colunas vetoriais (ignorar z, x, y que já temos)
            vector_columns = [
                "Na_x",
                "Na_y",
                "Na_z",
                "Nb_x",
                "Nb_y",
                "Nb_z",
                "U_x",
                "U_y",
                "U_z",
                "U_ver_x",
                "U_ver_y",
                "U_ver_z",
                "ja_x",
                "ja_y",
                "ja_z",
                "jb_x",  # Apenas jb_x disponível
            ]
            for col in vector_columns:
                if col in df_vector.columns:
                    all_data[col] = df_vector[col]
                    print(f"  ➕ Adicionado componente vetorial: {col}")
    else:
        print("⚠️  Arquivo de vetores não encontrado")

    # TERCEIRO: Identificar campos disponíveis para plotagem
    available_fields = []

    # Lista de todos os campos possíveis
    all_possible_fields = [
        "rho_a",
        "rho_b",
        "wa",
        "wb",
        "U",
        "U_ver",
        "ja",
        "jb",
        "Na",
        "Nb",
    ]

    for field in all_possible_fields:
        if field in ["U", "U_ver", "ja", "jb", "Na", "Nb"]:
            # Para campos vetoriais, verificar se temos pelo menos componente x
            if f"{field}_x" in all_data.columns:
                available_fields.append(field)
        else:
            # Para campos escalares
            if field in all_data.columns:
                available_fields.append(field)

    print(f"✅ Campos disponíveis para plotagem: {available_fields}")
    print("📋 Todas as colunas disponíveis:", list(all_data.columns))

    if all_data.empty:
        print("❌ Nenhum dado válido foi lido para plotagem ou exportação.")
        return

    # QUARTO: Calcular TODAS as soluções analíticas
    print("\n🧮 Calculando soluções analíticas para todos os campos...")

    # CORREÇÃO: Verificar se wa0 é válido
    if wa0 >= 1.0:
        print(f"⚠️  Aviso: wa0 = {wa0} é >= 1.0, ajustando para 0.999")
        wa0 = 0.999

    analytical_solutions = calculate_analytical_solution(
        all_data["z"], wa0, rho_total, Dab
    )

    # Adicionar soluções analíticas ao DataFrame
    for field, values in analytical_solutions.items():
        all_data[f"{field}_analytical"] = values

        # CORREÇÃO: Calcular erro absoluto para todos os campos
        numerical_field = field
        if field in ["U", "U_ver", "ja", "jb", "Na", "Nb"]:
            numerical_field = f"{field}_x"

        if numerical_field in all_data.columns:
            all_data[f"erro_absoluto_{field}"] = (
                all_data[numerical_field] - all_data[f"{field}_analytical"]
            )
            print(f"  ✅ Erro absoluto calculado para {field}")

    print(f"✅ Dados processados com sucesso: {len(all_data)} pontos")

    # Verificar se as soluções analíticas foram calculadas corretamente
    print("\n🔍 Verificando soluções analíticas:")
    for field in analytical_solutions.keys():
        analytical_col = f"{field}_analytical"
        if analytical_col in all_data.columns:
            print(f"  {analytical_col}: {all_data[analytical_col].head(3).tolist()}")

    # QUINTO: Exportar dados para CSV com 15 casas decimais
    csv_output_path = os.path.join(case_dir, f"sampled_data.csv")

    # Ordenar colunas para melhor organização
    coord_cols = [col for col in ["x", "y", "z"] if col in all_data.columns]
    numerical_cols = [
        col
        for col in all_data.columns
        if not col.endswith("_analytical")
        and not col.startswith("erro_absoluto")
        and col not in coord_cols
    ]
    analytical_cols = [col for col in all_data.columns if col.endswith("_analytical")]
    error_cols = [col for col in all_data.columns if col.startswith("erro_absoluto")]

    ordered_cols = (
        coord_cols
        + sorted(numerical_cols)
        + sorted(analytical_cols)
        + sorted(error_cols)
    )

    all_data[ordered_cols].to_csv(
        csv_output_path, index=False, sep=";", decimal=",", float_format="%.15f"
    )
    print(f"💾 Dados exportados para CSV (15 casas decimais): {csv_output_path}")

    # SEXTO: Processar campos para plotagem (individual ou combinada)
    process_fields_for_plotting(
        all_data, field_names, available_fields, plots_dir, wa0, rho_total, Dab
    )


def process_fields_for_plotting(
    data, requested_fields, available_fields, plots_dir, wa0, rho_total, Dab
):
    """
    Processa os campos para plotagem individual ou combinada.
    """
    print(f"\n🎨--- INICIANDO PLOTAGEM ---")
    print(f"📋 Campos solicitados: {requested_fields}")
    print(f"📊 Campos disponíveis: {available_fields}")

    # NOVA LÓGICA: Identificar grupos combinados diretamente dos campos solicitados
    individual_fields = []
    combined_groups = []

    # Primeiro, identificar todos os grupos combinados (strings sem espaços com múltiplos campos)
    for field_request in requested_fields:
        if len(field_request) > 2 and " " not in field_request:
            # Tentar dividir em campos de 2 letras (wa, wb, Na, Nb, etc.)
            possible_fields = []

            # Tentar dividir em pares de 2 caracteres
            for i in range(0, len(field_request), 2):
                if i + 2 <= len(field_request):
                    possible_field = field_request[i : i + 2]
                    possible_fields.append(possible_field)

            # Verificar quais desses campos estão disponíveis
            valid_fields = [f for f in possible_fields if f in available_fields]

            if len(valid_fields) >= 2:  # Pelo menos 2 campos válidos para combinar
                combined_groups.append(valid_fields)
                print(
                    f"  ✅ Grupo combinado identificado: '{field_request}' -> {valid_fields}"
                )
            else:
                print(
                    f"  ⚠️  Grupo '{field_request}' não pôde ser processado. Campos válidos: {valid_fields}"
                )
        else:
            # Campo individual
            if field_request in available_fields:
                individual_fields.append(field_request)
            else:
                print(f"  ⚠️  Campo individual '{field_request}' não disponível")

    print(f"  📈 Campos individuais: {individual_fields}")
    print(f"  📊 Grupos combinados: {combined_groups}")

    # Plotar campos individuais
    if individual_fields:
        plot_individual_fields(data, individual_fields, plots_dir, wa0, rho_total, Dab)

    # Plotar campos combinados
    for combined_fields in combined_groups:
        plot_combined_fields(data, combined_fields, plots_dir, wa0, rho_total, Dab)

    print(f"✅--- PLOTAGEM CONCLUÍDA ---")
    print(f"📁 Todos os plots salvos em: {plots_dir}")

    # Verificar se os arquivos foram realmente criados
    print(f"\n🔍 Verificando arquivos criados em {plots_dir}:")
    if os.path.exists(plots_dir):
        files = os.listdir(plots_dir)
        if files:
            for file in files:
                file_path = os.path.join(plots_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   📄 {file} ({file_size} bytes)")
        else:
            print("   ❌ Nenhum arquivo encontrado na pasta plots")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("❌ Erro: Especifique pelo menos um campo para processar.")
        print("💡 Uso: python3 postproc.py <campo1> [campo2] ...")
        print("📋 Campos disponíveis: rho_a, rho_b, wa, wb, U, U_ver, ja, jb, Na, Nb")
        print("🎯 Exemplo para campos individuais: python3 postproc.py wa wb U Na")
        print("🎯 Exemplo para campos combinados: python3 postproc.py wawb NaNb")
        print("💡 'wawb' cria UM gráfico com wa E wb juntos")
        print("💡 'wa wb' cria DOIS gráficos separados")
        sys.exit(1)

    case_directory = "."
    fields_to_process = sys.argv[1:]

    print(f"🔍 Processando caso no diretório atual: {os.path.abspath(case_directory)}")
    print(f"📋 Campos solicitados: {fields_to_process}")

    run_openfoam_postprocess(case_directory)
    plot_and_export_openfoam_data(case_directory, fields_to_process)
    print("🎉 Processo de automação completo concluído!")
