"""Experimento reprodutível da classificação Online Shoppers.

Uso, a partir da raiz do repositório:
    python docs/projects/classification/code/experiment.py search
    python docs/projects/classification/code/experiment.py final

`search` não calcula nenhum resultado de teste. `final` só deve ser executado depois
de congelar a configuração em search.json. O pré-processamento vem do EDA.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
EDA_CODE = PROJECTS / "eda" / "code"
sys.path.insert(0, str(EDA_CODE))
from preprocessing import MODEL_FEATURES, build_preprocessor  # noqa: E402

ROOT = HERE.parent
DATA = PROJECTS / "eda" / "data" / "online_shoppers_intention.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
SEED = 42
MAX_EPOCHS = 120
PATIENCE = 15
BATCH = 128
COLORS = {"mlp": "#2556a8", "logistic": "#d47722", "dummy": "#767b83"}
torch.set_num_threads(1)

# A grade foi fixada antes de consultar a validação. Há quatro estados de
# regularização: nenhuma, somente weight decay, somente dropout, e ambas.
CONFIGS = [
    {"id": "C1", "hidden": [32], "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.0},
    {"id": "C2", "hidden": [64], "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.0},
    {"id": "C3", "hidden": [64], "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.0},
    {"id": "C4", "hidden": [64], "lr": 1e-3, "weight_decay": 0.0, "dropout": 0.1},
    {"id": "C5", "hidden": [64], "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.1},
    {"id": "C6", "hidden": [64, 32], "lr": 1e-3, "weight_decay": 1e-3, "dropout": 0.1},
    {"id": "C7", "hidden": [128], "lr": 5e-4, "weight_decay": 1e-2, "dropout": 0.2},
    {"id": "C8", "hidden": [32, 16], "lr": 5e-4, "weight_decay": 1e-3, "dropout": 0.2},
]


def load_outer_split():
    """Reproduz exatamente a primeira dobra externa documentada no EDA."""
    frame = pd.read_csv(DATA)
    x_raw = frame.drop(columns="Revenue")
    y = frame["Revenue"].astype(int).to_numpy()
    groups = pd.util.hash_pandas_object(x_raw, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(x_raw, y, groups))
    assert (len(train_idx), len(test_idx)) == (9864, 2466)
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    return frame, train_idx, test_idx, groups


def load_inner_split(frame, outer_train_idx, groups):
    y_outer = frame.iloc[outer_train_idx]["Revenue"].astype(int).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    sub_idx, val_idx = next(
        splitter.split(frame.iloc[outer_train_idx][MODEL_FEATURES], y_outer, groups[outer_train_idx])
    )
    assert not (set(groups[outer_train_idx[sub_idx]]) & set(groups[outer_train_idx[val_idx]]))
    return outer_train_idx[sub_idx], outer_train_idx[val_idx]


def transform(frame, fit_idx, other_idx):
    """Ajusta o pipeline do EDA só no lado de treino da divisão recebida."""
    preprocess = build_preprocessor()
    x_fit = preprocess.fit_transform(frame.iloc[fit_idx][MODEL_FEATURES]).astype(np.float32)
    x_other = preprocess.transform(frame.iloc[other_idx][MODEL_FEATURES]).astype(np.float32)
    assert np.isfinite(x_fit).all() and np.isfinite(x_other).all()
    return preprocess, x_fit, x_other


class ShoppingMLP(nn.Module):
    """MLP própria: camadas configuráveis, ReLU e logits sem sigmoid final."""

    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        sizes = [input_dim] + hidden + [1]
        layers = []
        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu" if i < len(sizes) - 2 else "linear")
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def seed_torch(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_mlp(x_train, y_train, config, seed, *, x_val=None, y_val=None, epochs=MAX_EPOCHS):
    """AdamW + BCE ponderada. Validação opcional só para a etapa de busca."""
    seed_torch(seed)
    model = ShoppingMLP(x_train.shape[1], config["hidden"], config["dropout"])
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    positive_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    history = {"train_loss": [], "val_loss": [], "val_ap": []}
    best_loss, best_epoch, best_state, wait = float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history["train_loss"].append(running / len(y_train))
        if x_val is None:
            continue
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_val))
            val_loss = loss_fn(logits, torch.from_numpy(y_val.astype(np.float32))).item()
            val_prob = torch.sigmoid(logits).numpy()
        val_ap = average_precision_score(y_val, val_prob)
        history["val_loss"].append(val_loss)
        history["val_ap"].append(val_ap)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, best_state, wait = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if x_val is not None:
        model.load_state_dict(best_state)  # nunca usar o último iterando às cegas
    return model, history, best_epoch or epochs, best_loss


def probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def choose_threshold(y, probabilities_):
    # O limiar maximiza F1 positivo na validação; nunca é recalculado no teste.
    precision, recall, thresholds = precision_recall_curve(y, probabilities_)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))]), float(np.max(f1))


def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    p, r, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "ap": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "by_class": {str(k): {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f1[k]), "support": int(support[k])} for k in (0, 1)},
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def search():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame, train_idx, _, groups = load_outer_split()
    sub_idx, val_idx = load_inner_split(frame, train_idx, groups)
    preprocess, x_sub, x_val = transform(frame, sub_idx, val_idx)
    y_sub = frame.iloc[sub_idx]["Revenue"].astype(int).to_numpy()
    y_val = frame.iloc[val_idx]["Revenue"].astype(int).to_numpy()
    baselines = {}
    for name, estimator in [
        ("dummy", DummyClassifier(strategy="most_frequent")),
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]:
        estimator.fit(x_sub, y_sub)
        prob = estimator.predict_proba(x_val)[:, 1]
        threshold, _ = choose_threshold(y_val, prob) if name != "dummy" else (0.5, 0.0)
        baselines[name] = evaluate(y_val, prob, threshold)
    rows = []
    histories = {}
    for config in CONFIGS:
        model, history, best_epoch, best_loss = train_mlp(x_sub, y_sub, config, SEED, x_val=x_val, y_val=y_val)
        prob = probabilities(model, x_val)
        threshold, f1_val = choose_threshold(y_val, prob)
        result = {**config, "parameters": parameter_count(model), "best_epoch": best_epoch,
                  "epochs_run": len(history["train_loss"]), "best_val_loss": best_loss,
                  "val_ap": float(average_precision_score(y_val, prob)),
                  "val_f1_positive": f1_val, "threshold": threshold}
        rows.append(result)
        histories[config["id"]] = history
        print(f'{config["id"]}: AP validação={result["val_ap"]:.4f}, melhor época={best_epoch}', flush=True)
    rows.sort(key=lambda r: (-r["val_ap"], r["best_val_loss"], r["parameters"]))
    outcome = {
        "protocol": "PROTOCOL.md", "seed": SEED, "outer_train_size": len(train_idx),
        "subtrain_size": len(sub_idx), "validation_size": len(val_idx),
        "subtrain_positive": int(y_sub.sum()), "validation_positive": int(y_val.sum()),
        "input_dim_search": int(x_sub.shape[1]), "feature_names_search": preprocess.get_feature_names_out().tolist(),
        "baselines_validation": baselines, "configurations_ranked": rows,
        "selected_id": rows[0]["id"], "histories": histories,
    }
    save_json(OUTPUTS / "search.json", outcome)
    print("Configuração selecionada:", rows[0]["id"], flush=True)


def subgroup_errors(frame, indices, y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    category = frame.iloc[indices]["VisitorType"].astype(str).to_numpy()
    rows = []
    for name in sorted(set(category)):
        mask = category == name
        positives = int(y[mask].sum())
        negatives = int(mask.sum() - positives)
        fn = int(((y == 1) & (pred == 0) & mask).sum())
        fp = int(((y == 0) & (pred == 1) & mask).sum())
        rows.append({"group": name, "n": int(mask.sum()), "positives": positives,
                     "errors": fn + fp, "error_rate": float((fn + fp) / mask.sum()),
                     "false_negative_rate": float(fn / positives) if positives else None,
                     "false_positive_rate": float(fp / negatives) if negatives else None})
    return rows


def worst_cases(indices, y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    rows = []
    for kind, mask, rank in [
        ("false_positive", (y == 0) & (pred == 1), prob),
        ("false_negative", (y == 1) & (pred == 0), 1 - prob),
    ]:
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(rank[positions])[::-1][:3]]
        rows.extend({"kind": kind, "csv_row_zero_based": int(indices[i]), "true_label": int(y[i]),
                     "predicted_probability": float(prob[i])} for i in positions)
    return rows


def permutation_importance(frame, sub_idx, val_idx, config, epochs, seed):
    """Importância no conjunto de validação, nunca no teste."""
    preprocess, x_sub, x_val = transform(frame, sub_idx, val_idx)
    y_sub = frame.iloc[sub_idx]["Revenue"].astype(int).to_numpy()
    y_val = frame.iloc[val_idx]["Revenue"].astype(int).to_numpy()
    model, _, _, _ = train_mlp(x_sub, y_sub, config, seed, epochs=epochs)
    reference = float(average_precision_score(y_val, probabilities(model, x_val)))
    raw_val = frame.iloc[val_idx][MODEL_FEATURES].copy()
    rng = np.random.default_rng(SEED)
    rows = []
    for column in MODEL_FEATURES:
        shuffled = raw_val.copy()
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
        x_perm = preprocess.transform(shuffled).astype(np.float32)
        ap = float(average_precision_score(y_val, probabilities(model, x_perm)))
        rows.append({"feature": column, "ap_drop": reference - ap})
    rows.sort(key=lambda r: r["ap_drop"], reverse=True)
    return {"reference_validation_ap": reference, "rows": rows}


def final():
    search_result = json.loads((OUTPUTS / "search.json").read_text(encoding="utf-8"))
    selected = search_result["configurations_ranked"][0]
    assert selected["id"] == search_result["selected_id"]
    frame, train_idx, test_idx, groups = load_outer_split()
    sub_idx, val_idx = load_inner_split(frame, train_idx, groups)
    preprocess, x_train, x_test = transform(frame, train_idx, test_idx)
    y_train = frame.iloc[train_idx]["Revenue"].astype(int).to_numpy()
    y_test = frame.iloc[test_idx]["Revenue"].astype(int).to_numpy()
    baselines = {}
    curves = {}
    for name, estimator in [
        ("dummy", DummyClassifier(strategy="most_frequent")),
        ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]:
        estimator.fit(x_train, y_train)
        prob = estimator.predict_proba(x_test)[:, 1]
        threshold = search_result["baselines_validation"][name]["threshold"]
        baselines[name] = evaluate(y_test, prob, threshold)
        curves[name] = prob.tolist()
    mlp_runs = []
    first_prob = None
    for seed in (42, 43, 44):
        model, _, _, _ = train_mlp(x_train, y_train, selected, seed, epochs=selected["best_epoch"])
        prob = probabilities(model, x_test)
        metrics = evaluate(y_test, prob, selected["threshold"])
        mlp_runs.append({"seed": seed, "metrics": metrics})
        print(f'Semente {seed}: AP teste={metrics["ap"]:.4f}', flush=True)
        if first_prob is None:
            first_prob = prob
    importance = permutation_importance(frame, sub_idx, val_idx, selected, selected["best_epoch"], SEED)
    first_metrics = mlp_runs[0]["metrics"]
    aps = [run["metrics"]["ap"] for run in mlp_runs]
    # O comparador também é escolhido na validação, nunca pelo teste.
    best_baseline_name = max(
        search_result["baselines_validation"],
        key=lambda k: search_result["baselines_validation"][k]["ap"],
    )
    result = {
        "selected_id": selected["id"], "selected_config": selected,
        "outer_train_size": len(train_idx), "test_size": len(test_idx),
        "test_positive": int(y_test.sum()), "input_dim_final": int(x_train.shape[1]),
        "feature_names_final": preprocess.get_feature_names_out().tolist(),
        "baselines_test": baselines, "mlp_runs": mlp_runs,
        "mlp_ap_mean": float(np.mean(aps)), "mlp_ap_sd": float(np.std(aps, ddof=1)),
        "best_baseline_name": best_baseline_name,
        "best_baseline_ap": baselines[best_baseline_name]["ap"],
        "success_margin": float(np.mean(aps) - baselines[best_baseline_name]["ap"]),
        "success_reached": bool(np.mean(aps) - baselines[best_baseline_name]["ap"] >= 0.02),
        "subgroups": subgroup_errors(frame, test_idx, y_test, first_prob, selected["threshold"]),
        "worst_cases": worst_cases(test_idx, y_test, first_prob, selected["threshold"]),
        "permutation_importance": importance,
    }
    final_path = OUTPUTS / "final.json"
    if final_path.exists():
        # Num clone, o resultado original já foi entregue. Uma reprodução não
        # pode reescrever silenciosamente a evidência; deve conferir a AP.
        original = json.loads(final_path.read_text(encoding="utf-8"))
        observed = [run["metrics"]["ap"] for run in result["mlp_runs"]]
        expected = [run["metrics"]["ap"] for run in original["mlp_runs"]]
        if not np.allclose(observed, expected, atol=1e-4, rtol=0):
            raise RuntimeError(f"Reprodução divergente da AP publicada: {observed} vs. {expected}")
        if not np.isclose(result["best_baseline_ap"], original["best_baseline_ap"], atol=1e-4, rtol=0):
            raise RuntimeError("Reprodução divergente da AP do baseline")
        print("Reprodução compatível com final.json; evidência original preservada.", flush=True)
    else:
        save_json(final_path, result)
    make_figures(search_result, result, y_test, first_prob, curves)


def make_figures(search_result, result, y_test, mlp_prob, curves):
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    selected = result["selected_config"]
    hist = search_result["histories"][selected["id"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], label="Subtreino", color=COLORS["mlp"])
    ax.plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label="Validação", color=COLORS["logistic"])
    ax.axvline(selected["best_epoch"], color="#444444", linestyle="--", label=f'Melhor época: {selected["best_epoch"]}')
    ax.set(title="Figura 1 — Perda BCE ponderada por época", xlabel="Época", ylabel="BCE ponderada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_01_training.png", dpi=160)
    plt.close(fig)

    ranked = search_result["configurations_ranked"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar([r["id"] for r in ranked], [r["val_ap"] for r in ranked], color=COLORS["mlp"], label="AP na validação")
    ax.axhline(search_result["baselines_validation"]["logistic"]["ap"], color=COLORS["logistic"], linestyle="--", label="Regressão logística")
    ax.set(title="Figura 2 — Busca de oito configurações", xlabel="Configuração (ordenada por AP)", ylabel="Average precision (AP)", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_02_search.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, prob in [("MLP", mlp_prob), ("Logística", curves["logistic"])]:
        p, r, _ = precision_recall_curve(y_test, prob)
        key = {"MLP": "mlp", "Logística": "logistic"}[name]
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_test, prob):.3f})", color=COLORS[key])
    # Um classificador de escore constante não gera uma curva operacional:
    # precision_recall_curve devolve só extremos, cuja interpolação seria falsa.
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, linestyle="--", label=f"Trivial: prevalência={prevalence:.3f}", color=COLORS["dummy"])
    ax.set(title="Figura 3 — Precisão–recall no teste", xlabel="Recall da classe compra", ylabel="Precisão da classe compra", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_03_pr.png", dpi=160)
    plt.close(fig)

    matrix = np.asarray(result["mlp_runs"][0]["metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set(title="Figura 4 — Matriz de confusão no teste\n(semente 42)", xlabel="Classe prevista", ylabel="Classe verdadeira", xticks=[0, 1], yticks=[0, 1])
    ax.set_xticklabels(["Não compra", "Compra"])
    ax.set_yticklabels(["Não compra", "Compra"])
    fig.colorbar(image, ax=ax, label="Número de sessões")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_04_confusion.png", dpi=160)
    plt.close(fig)

    subgroup = result["subgroups"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([r["group"] for r in subgroup], [100 * r["error_rate"] for r in subgroup], color=COLORS["mlp"], label="Erros / sessões")
    ax.set(title="Figura 5 — Taxa de erro por tipo de visitante (teste)", xlabel="Tipo de visitante", ylabel="Taxa de erro (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_05_subgroups.png", dpi=160)
    plt.close(fig)

    top = result["permutation_importance"]["rows"][:10]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([r["feature"] for r in reversed(top)], [r["ap_drop"] for r in reversed(top)], color=COLORS["mlp"], label="Queda de AP")
    ax.set(title="Figura 6 — Importância por permutação na validação", xlabel="AP original − AP após permutação", ylabel="Feature bruta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_06_importance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["search", "final"])
    args = parser.parse_args()
    search() if args.phase == "search" else final()
