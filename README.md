# openfoam2012-projects

# OpenFOAM 2012 - Desenvolvimento

## 📋 Status do Projeto
- **Última Reunião**: 16/out/2025
- **Próxima Reunião**: 23/out/2025 
- **Branch Ativa**: `feature/correcoes-out-16`

## 🎯 Tarefas Atuais
- [ ] Gŕafico 3 condições de contoro (0.5, 0.75, 0.95)
- [ ] Gráficos de erros absolutos STF x solvers
    - [x] script para retirar U de equimolarDiffusionFoam -> 0/U pra RSTF
    - [x] rodar os casos no RSTF
- [ ] Estudo de malha
- [x] Criar variável U_ver
    - [x] U pelo somatório dos fluxos
    - [x] Separar de U calculada pela reorganização da eq (U em função de ja)
- [x] Atualizar scripts de pós processamento
- [x] Criação do git


## 📁 Estrutura
- `solvers/` - Desenvolvimento de solvers
- `run/` - Casos de simulação
- `docs/` - Registros e documentação
- `scripts/` - Scripts de gráfico e pós processamento