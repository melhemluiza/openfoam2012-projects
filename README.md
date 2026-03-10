# openfoam2012-projects

# OpenFOAM 2012 - Projetos e Desenvolvimento

## 📋 Sobre o Projeto

Repositório dedicado ao desenvolvimento e validação de solvers personalizados no OpenFOAM v2012, com foco em problemas de difusão e transferência de massa. O projeto inclui a implementação de solvers, casos de teste, scripts de pós-processamento e documentação do progresso.
Analisamos dois cenários: difusão de A em B estagnado (1) e contra-difusão equimolar (2).
## 🔬 Metodologia de Trabalho

### Casos em Desenvolvimento
- *Difusão com diferentes condições de contorno*: Análise para valores 0.5, 0.75 e 0.95
- *Comparação de solvers*: STF (Scalar Transport Foam) vs solvers personalizados
- *Estudos de malha*: Verificação de convergência e independência de malha
- *Implementação de variáveis*: Criação de U_ver para diferentes abordagens de cálculo

## 🛠️ Scripts de Pós-processamento

### Scripts em Desenvolvimento
- *grafico.py* - Geração de gráficos para diferentes condições
- *setup_U.py* - Extração de U do equimolarDiffusionFoam para uso no RSTF
- *postproc.py* - Scripts de pós-processamento em atualização

## 📁 Estrutura do Projeto
openfoam2012-projects/
├── solvers/ # Desenvolvimento de solvers
│ ├── binaryDiffusionFoam/
│ ├── equimolarDiffusionFoam/
│ └── (outros solvers)
├── run/ # Casos de simulação
│ ├── Test1  (BDF caso 1)
│ ├── Tesr2  (STF caso 1)
│ ├── Test3  (EDF caso 2)
│ └── Test4  (STF caso 2)
├── scripts/
│ ├── grafico.py
│ ├── erros_absolutos.py
│ ├── setup_U.py
│ └── postproc.py
└── docs/
└── reunioes/ # Acompanhamento semanal

## 🔄 Estratégia de Branches

### Branches por Reunião/Atualização
feature/YYYY-MM-DD-descricao-breve

### O que cada branch guarda:
- Versão atual dos solvers naquela data
- Modificações nos scripts
- Resultados e análises do período
- Documentação do progresso e correções

## ✅ Status Atual do Projeto

### 📅 Última Reunião: *13/nov/2025*
### 📅 Próxima Reunião: *27/nov/2025*

### ✅ *VALIDADO*
- Test1 vs Test2 (binaryDiffusionFoam ≡ rhoscalarTransportFoam)
- Test3 vs Test4 (equimolarDiffusionFoam ≡ rhoscalarTransportFoam)
- Test5 e Test6 vs Tes2 e Test4 (fluxTransportFoam ≡ rhoscalarTransportFoam - um pouco pior)

### 🔄 *EM ANDAMENTO*


### ✅ *CONCLUÍDO*
- [x] Criação do repositório
- [x] Configuração inicial da estrutura
- [x] Gráfico para 3 condições de contorno (0.5, 0.75, 0.95)
- [x] Gráficos de erros absolutos STF x solvers
- [x] Script para extrair U de equimolarDiffusionFoam (0/U) para RSTF
- [x] Executar casos no RSTF
- [x] Estudo de malha
- [x] Criar variável U_ver:
  - [x] U pelo somatório dos fluxos
  - [x] Separar de U calculada pela reorganização da eq (U em função de ja)
- [x] Atualizar scripts de pós-processamento

