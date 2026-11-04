# EDA — Redes Neurais e Deep Learning (Insper, 2026.2)

Repositório do projeto individual de Sérgio Carmelo Tôrres Filho, criado a partir do [template da disciplina](https://github.com/repo-classes/ann-dl). O [relatório do EDA](docs/projects/eda/index.md) explora o [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) e prepara um pipeline para classificação binária de `Revenue`. O integrante informou em 04/11/2026 que o professor aprovou a base. Nenhum modelo é treinado nesta etapa.

## Reproduzir

Use Python 3.11 ou superior, a partir da raiz do repositório:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python docs/projects/eda/code/eda_analysis.py
.venv\Scripts\python docs/projects/eda/code/test_preprocessing.py
.venv\Scripts\mkdocs build --strict
```

Em Linux/macOS, ative a `.venv` e substitua os caminhos `.venv\Scripts\` por `.venv/bin/`. O script lê o [CSV versionado](docs/projects/eda/data/online_shoppers_intention.csv), usa `random_state=42`, gera as figuras em `docs/projects/eda/figures/` e as tabelas/medidas em `docs/projects/eda/outputs/`. O pipeline importável está em [preprocessing.py](docs/projects/eda/code/preprocessing.py). Não depende de download durante a execução.

## Fonte e licença

Sakar & Kastro (2018), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository, [DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A cópia do CSV tem SHA-256 `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
