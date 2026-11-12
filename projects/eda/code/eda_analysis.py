"""EDA reproduzível: execute ``python docs/projects/eda/code/eda_analysis.py``.

Não treina modelo. Gera tabelas, 11 figuras e um JSON com os números do relatório.
Os estimadores de pré-processamento e PCA veem apenas as linhas de treino.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from umap import UMAP

from preprocessing import (
    BOUNDED_NUMERIC,
    CATEGORICAL,
    EXCLUDED,
    MODEL_FEATURES,
    SKEWED_NUMERIC,
    build_preprocessor,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "online_shoppers_intention.csv"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
ALL_NUMERIC = SKEWED_NUMERIC + BOUNDED_NUMERIC + ["PageValues"]
CLASS_COLORS = {False: "#315f91", True: "#d48935"}
CLASS_NAMES = {False: "Sem compra", True: "Compra"}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown_table(filename: str, headers: list[str], rows: list[list[str]]) -> None:
    """Gera uma tabela inspecionável que o relatório inclui sem copiá-la à mão."""
    content = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    content.extend("| " + " | ".join(row) + " |" for row in rows)
    (OUTPUTS / filename).write_text("\n".join(content) + "\n", encoding="utf-8")


def class_histogram(ax, frame: pd.DataFrame, column: str, log1p: bool) -> None:
    for label in (False, True):
        values = frame.loc[frame["Revenue"] == label, column].astype(float)
        if log1p:
            values = np.log1p(values)
        ax.hist(
            values,
            bins=35,
            density=True,
            alpha=0.50,
            color=CLASS_COLORS[label],
            label=CLASS_NAMES[label],
        )
    ax.set_title(column)
    ax.set_xlabel(f"log(1 + {column})" if log1p else column)
    ax.set_ylabel("Densidade")
    ax.legend(fontsize=8)


def class_scatter(ax, coordinates, labels, title: str, x_name: str, y_name: str) -> None:
    for label in (False, True):
        selected = labels == label
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=12,
            alpha=0.55,
            c=CLASS_COLORS[label],
            label=CLASS_NAMES[label],
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.legend(markerscale=1.5, fontsize=8)


def plot_target(frame: pd.DataFrame) -> None:
    counts = frame["Revenue"].value_counts().reindex([False, True])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for label in (False, True):
        ax.bar(CLASS_NAMES[label], counts[label], color=CLASS_COLORS[label], label=CLASS_NAMES[label])
        ax.text(CLASS_NAMES[label], counts[label] + 80, f"{counts[label]:,}", ha="center")
    ax.set_title("Figura 1 — Desfecho das sessões (dataset completo)")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("Número de sessões")
    ax.set_ylim(0, counts.max() * 1.13)
    ax.legend(title="Classe")
    save_figure(fig, "figure_01_target.png")


def plot_univariate(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for ax, column in zip(axes.flat, SKEWED_NUMERIC):
        class_histogram(ax, train, column, log1p=True)
    fig.suptitle("Figura 2 — Contagens e durações: distribuição no treino", fontsize=14)
    save_figure(fig, "figure_02_numeric_skewed.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, column in zip(axes.flat, BOUNDED_NUMERIC + ["PageValues"]):
        class_histogram(ax, train, column, log1p=column == "PageValues")
    fig.suptitle("Figura 3 — Taxas, proximidade de feriados e PageValues no treino", fontsize=14)
    save_figure(fig, "figure_03_numeric_other.png")


def categorical_order(frame: pd.DataFrame, column: str) -> list:
    if column == "Month":
        present = set(frame[column].astype(str))
        return [month for month in MONTH_ORDER if month in present]
    return frame[column].value_counts().index.tolist()


def plot_categorical(train: pd.DataFrame) -> None:
    chosen = ["Month", "OperatingSystems", "VisitorType", "Weekend"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, column in zip(axes.flat, chosen):
        order = categorical_order(train, column)
        counts = pd.crosstab(train[column], train["Revenue"]).reindex(order, fill_value=0)
        lower = counts.get(False, pd.Series(0, index=order))
        upper = counts.get(True, pd.Series(0, index=order))
        positions = np.arange(len(order))
        ax.bar(positions, lower, color=CLASS_COLORS[False], label=CLASS_NAMES[False])
        ax.bar(positions, upper, bottom=lower, color=CLASS_COLORS[True], label=CLASS_NAMES[True])
        ax.set_xticks(positions, [str(value) for value in order], rotation=40 if len(order) > 7 else 0)
        ax.set_title(column)
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Sessões")
        ax.legend(fontsize=8)
    fig.suptitle("Figura 4 — Frequências categóricas no treino", fontsize=14)
    save_figure(fig, "figure_04_categorical.png")


def plot_numeric_relationships(train: pd.DataFrame, correlations: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    sns.heatmap(
        correlations,
        ax=ax,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Correlação de Spearman"},
    )
    ax.set_title("Figura 5 — Correlação entre numéricas no treino")
    ax.set_xlabel("Feature numérica")
    ax.set_ylabel("Feature numérica")
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    save_figure(fig, "figure_05_spearman.png")

    sample = train.sample(n=min(3000, len(train)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    comparisons = [
        ("Informational", "Informational_Duration", "Páginas informativas", "log(1 + duração informativa)"),
        ("BounceRates", "ExitRates", "BounceRates (proporção)", "ExitRates (proporção)"),
    ]
    for ax, (x_column, y_column, x_label, y_label) in zip(axes, comparisons):
        for label in (False, True):
            selected = sample["Revenue"] == label
            y_values = sample.loc[selected, y_column]
            if y_column == "Informational_Duration":
                y_values = np.log1p(y_values)
            ax.scatter(
                sample.loc[selected, x_column],
                y_values,
                alpha=0.35,
                s=12,
                color=CLASS_COLORS[label],
                label=CLASS_NAMES[label],
            )
        ax.set_title(f"{x_column} × {y_column}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(title="Revenue")
    fig.suptitle("Figura 6 — Pares numéricos associados (amostra do treino)", fontsize=14)
    save_figure(fig, "figure_06_bounce_exit.png")


def plot_categorical_target(train: pd.DataFrame) -> None:
    chosen = ["Month", "VisitorType", "Weekend"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, column in zip(axes, chosen):
        order = categorical_order(train, column)
        proportions = pd.crosstab(train[column], train["Revenue"], normalize="index").reindex(order, fill_value=0)
        negative = proportions.get(False, pd.Series(0.0, index=order)) * 100
        positive = proportions.get(True, pd.Series(0.0, index=order)) * 100
        x = np.arange(len(order))
        ax.bar(x, negative, color=CLASS_COLORS[False], label=CLASS_NAMES[False])
        ax.bar(x, positive, bottom=negative, color=CLASS_COLORS[True], label=CLASS_NAMES[True])
        ax.set_xticks(x, [str(value) for value in order], rotation=45 if len(order) > 7 else 0)
        ax.set_title(column)
        ax.set_xlabel("Categoria")
        ax.set_ylabel("Proporção de sessões (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
    fig.suptitle("Figura 7 — Composição de Revenue por categoria (treino)", fontsize=14)
    save_figure(fig, "figure_07_category_target.png")


def plot_numeric_categorical(train: pd.DataFrame) -> None:
    frame = train.copy()
    frame["Revenue_label"] = frame["Revenue"].map(CLASS_NAMES)
    frame["ProductRelated_Duration_log"] = np.log1p(frame["ProductRelated_Duration"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    sns.boxplot(
        data=frame,
        x="VisitorType",
        y="ProductRelated_Duration_log",
        hue="Revenue_label",
        hue_order=[CLASS_NAMES[False], CLASS_NAMES[True]],
        palette=[CLASS_COLORS[False], CLASS_COLORS[True]],
        showfliers=False,
        ax=axes[0],
    )
    axes[0].set_title("Tempo em páginas de produto × visitante")
    axes[0].set_xlabel("VisitorType")
    axes[0].set_ylabel("log(1 + ProductRelated_Duration)")
    axes[0].legend(title="Revenue", fontsize=8)
    sns.boxplot(
        data=frame,
        x="Weekend",
        y="ExitRates",
        hue="Revenue_label",
        hue_order=[CLASS_NAMES[False], CLASS_NAMES[True]],
        palette=[CLASS_COLORS[False], CLASS_COLORS[True]],
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title("Taxa de saída × fim de semana")
    axes[1].set_xlabel("Weekend")
    axes[1].set_ylabel("ExitRates (proporção)")
    axes[1].legend(title="Revenue", fontsize=8)
    fig.suptitle("Figura 8 — Numéricas × categóricas no treino (outliers não desenhados)", fontsize=14)
    save_figure(fig, "figure_08_numeric_category.png")


def plot_projections(train_t, labels, feature_names, output) -> dict:
    # PCA é ajustada no treino inteiro; as três dispersões usam a MESMA amostra.
    pca = PCA(n_components=10, random_state=42)
    pca.fit(train_t)
    explained = pca.explained_variance_ratio_
    scores = pca.transform(train_t)
    sample_idx, _ = train_test_split(
        np.arange(len(labels)), train_size=1500, stratify=labels, random_state=42
    )
    sample_x = train_t[sample_idx]
    sample_y = np.asarray(labels)[sample_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    class_scatter(axes[0], scores[sample_idx, :2], sample_y, "PCA: amostra estratificada do treino", "PC1", "PC2")
    axes[1].plot(np.arange(1, 11), np.cumsum(explained) * 100, marker="o", color="#315f91", label="Variância acumulada")
    axes[1].set_title("Variância explicada acumulada")
    axes[1].set_xlabel("Número de componentes")
    axes[1].set_ylabel("Variância explicada (%)")
    axes[1].set_xticks(np.arange(1, 11))
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    fig.suptitle("Figura 9 — PCA nas features transformadas do treino", fontsize=14)
    save_figure(fig, "figure_09_pca.png")

    loadings = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=["PC1", "PC2"])
    loadings.to_csv(output / "pca_loadings.csv", index_label="feature")

    tsne_settings = [30, 80]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, perplexity in zip(axes, tsne_settings):
        embedded = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=42,
        ).fit_transform(sample_x)
        class_scatter(ax, embedded, sample_y, f"Perplexity = {perplexity}", "t-SNE 1", "t-SNE 2")
    fig.suptitle("Figura 10 — t-SNE na mesma amostra do treino", fontsize=14)
    save_figure(fig, "figure_10_tsne.png")

    umap_settings = [15, 50]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, neighbors in zip(axes, umap_settings):
        embedded = UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=0.1,
            random_state=42,
            n_jobs=1,
        ).fit_transform(sample_x)
        class_scatter(ax, embedded, sample_y, f"n_neighbors = {neighbors}", "UMAP 1", "UMAP 2")
    fig.suptitle("Figura 11 — UMAP na mesma amostra do treino", fontsize=14)
    save_figure(fig, "figure_11_umap.png")

    top_loadings = {}
    for component in ["PC1", "PC2"]:
        ordered = loadings[component].abs().sort_values(ascending=False).head(5)
        top_loadings[component] = [
            {"feature": name, "loading": float(loadings.loc[name, component])}
            for name in ordered.index
        ]
    return {
        "sample_size": len(sample_idx),
        "sample_class_counts": np.bincount(sample_y.astype(int)).tolist(),
        "explained_variance_ratio": explained.tolist(),
        "pc1_pc2_variance_pct": float(explained[:2].sum() * 100),
        "top_loadings": top_loadings,
        "tsne_perplexities": tsne_settings,
        "umap_n_neighbors": umap_settings,
    }


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", palette=[CLASS_COLORS[False], CLASS_COLORS[True]])
    raw_bytes = DATA.read_bytes()
    frame = pd.read_csv(DATA)
    assert frame.shape == (12330, 18)
    assert set(frame["Revenue"].unique()) == {False, True}

    # O agrupamento por assinatura impede cópias exatas nos dois lados do split.
    x_all = frame.drop(columns="Revenue")
    labels = frame["Revenue"].astype(int)
    signatures = pd.util.hash_pandas_object(x_all, index=False).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(x_all, labels, signatures))
    train = frame.iloc[train_idx].copy()
    test = frame.iloc[test_idx].copy()
    assert not (set(signatures[train_idx]) & set(signatures[test_idx]))

    # Inspeção bruta, inclusive itens que não entrarão no modelo.
    missing = pd.DataFrame({
        "column": frame.columns,
        "missing_count": frame.isna().sum().to_numpy(),
        "missing_pct": (frame.isna().mean() * 100).to_numpy(),
    })
    missing.to_csv(OUTPUTS / "missing_values.csv", index=False)
    invalid = {
        "negative_counts_or_durations": int((frame[SKEWED_NUMERIC] < 0).to_numpy().sum()),
        "negative_page_values": int((frame["PageValues"] < 0).sum()),
        "rates_outside_0_1": int(((frame[BOUNDED_NUMERIC] < 0) | (frame[BOUNDED_NUMERIC] > 1)).to_numpy().sum()),
        "nonpositive_categorical_codes": int((frame[["OperatingSystems", "Browser", "Region", "TrafficType"]] <= 0).to_numpy().sum()),
        "unknown_months": int((~frame["Month"].isin(MONTH_ORDER)).sum()),
        "blank_string_cells": int(sum(frame[c].astype(str).str.strip().eq("").sum() for c in ["Month", "VisitorType"])),
    }

    # Todas as tabelas descritivas seguintes usam somente treino.
    numeric = train[ALL_NUMERIC].describe().T
    numeric["median"] = train[ALL_NUMERIC].median()
    numeric["zero_pct"] = train[ALL_NUMERIC].eq(0).mean() * 100
    numeric["skewness"] = train[ALL_NUMERIC].skew()
    numeric["p99"] = train[ALL_NUMERIC].quantile(0.99)
    numeric.to_csv(OUTPUTS / "numeric_summary.csv", index_label="feature")

    categorical_rows = []
    categorical_summary = {}
    for column in CATEGORICAL:
        counts = train[column].value_counts(dropna=False)
        rare = counts[counts / len(train) < 0.01]
        categorical_summary[column] = {
            "cardinality": int(train[column].nunique(dropna=False)),
            "most_common": str(counts.index[0]),
            "most_common_count": int(counts.iloc[0]),
            "most_common_pct": float(counts.iloc[0] / len(train) * 100),
            "rare_categories_below_1pct": [str(v) for v in rare.index],
        }
        for value, count in counts.items():
            categorical_rows.append({
                "feature": column,
                "category": str(value),
                "count": int(count),
                "pct": float(count / len(train) * 100),
            })
    pd.DataFrame(categorical_rows).to_csv(OUTPUTS / "categorical_frequencies.csv", index=False)
    save_markdown_table(
        "categorical_frequencies.txt",
        ["Feature", "Categoria", "Sessões", "% do treino"],
        [
            [row["feature"], row["category"], str(row["count"]), f'{row["pct"]:.2f}'.replace(".", ",")]
            for row in categorical_rows
        ],
    )

    correlations = train[ALL_NUMERIC].corr(method="spearman")
    correlations.to_csv(OUTPUTS / "spearman_correlations.csv", index_label="feature")
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    top_pair = upper.abs().stack().idxmax()
    category_target = {}
    for column in CATEGORICAL:
        rates = train.groupby(column, dropna=False)["Revenue"].agg(["size", "mean"])
        category_target[column] = [
            {"category": str(index), "count": int(row["size"]), "purchase_pct": float(row["mean"] * 100)}
            for index, row in rates.iterrows()
        ]
    pd.DataFrame(
        [dict(feature=column, **row) for column, rows in category_target.items() for row in rows]
    ).to_csv(OUTPUTS / "category_target_rates.csv", index=False)

    # O pipeline é importado do módulo vizinho; nenhuma estatística usa o teste no fit.
    preprocess = build_preprocessor()
    x_train_t = preprocess.fit_transform(train[MODEL_FEATURES])
    x_test_t = preprocess.transform(test[MODEL_FEATURES])
    assert np.isfinite(x_train_t).all() and np.isfinite(x_test_t).all()
    names = preprocess.get_feature_names_out()
    pd.DataFrame({"feature_name": names}).to_csv(OUTPUTS / "feature_names.csv", index=False)
    save_markdown_table(
        "feature_names.txt",
        ["#", "Feature após o pipeline"],
        [[str(index), f"`{name}`"] for index, name in enumerate(names, start=1)],
    )
    caps = preprocess.named_transformers_["skew"].named_steps["cap"].upper_
    train_skewed = train[SKEWED_NUMERIC].to_numpy(dtype=float)
    cap_flags = train_skewed > caps
    outlier_rows = int(np.any(cap_flags, axis=1).sum())
    cap_details = {name: {"cap": float(cap), "affected_rows": int(cap_flags[:, i].sum())}
                   for i, (name, cap) in enumerate(zip(SKEWED_NUMERIC, caps))}

    plot_target(frame)
    plot_univariate(train)
    plot_categorical(train)
    plot_numeric_relationships(train, correlations)
    plot_categorical_target(train)
    plot_numeric_categorical(train)
    projection = plot_projections(x_train_t, train["Revenue"].to_numpy(), names, OUTPUTS)

    result = {
        "source_url": "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
        "source_csv_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_shape": list(frame.shape),
        "raw_numeric_features": len(ALL_NUMERIC),
        "raw_categorical_features": len(CATEGORICAL),
        "raw_target_counts": {str(k): int(v) for k, v in frame["Revenue"].value_counts().items()},
        "raw_missing_total": int(frame.isna().to_numpy().sum()),
        "raw_exact_duplicate_rows": int(frame.duplicated().sum()),
        "raw_duplicate_feature_rows": int(x_all.duplicated().sum()),
        "invalid_counts": invalid,
        "excluded_features": EXCLUDED,
        "split": {
            "strategy": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), first fold",
            "train_size": len(train),
            "test_size": len(test),
            "train_positive": int(train["Revenue"].sum()),
            "test_positive": int(test["Revenue"].sum()),
            "shared_signatures": 0,
        },
        "categorical_summary": categorical_summary,
        "most_correlated_numeric_pair": {
            "features": list(top_pair),
            "spearman_rho": float(correlations.loc[top_pair]),
        },
        "bounce_exit_spearman_rho": float(correlations.loc["BounceRates", "ExitRates"]),
        "category_target_rates": category_target,
        "page_values_by_target": {
            str(label): {
                "mean": float(train.loc[train["Revenue"] == label, "PageValues"].mean()),
                "median": float(train.loc[train["Revenue"] == label, "PageValues"].median()),
            }
            for label in [False, True]
        },
        "outlier_capping": {
            "quantile": 0.99,
            "train_rows_affected": outlier_rows,
            "per_feature": cap_details,
        },
        "pipeline": {
            "train_shape": list(x_train_t.shape),
            "test_shape": list(x_test_t.shape),
            "feature_names": names.tolist(),
            "train_nan": int(np.isnan(x_train_t).sum()),
            "test_nan": int(np.isnan(x_test_t).sum()),
        },
        "projection": projection,
    }
    (OUTPUTS / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {key: result[key] for key in [
        "raw_shape", "raw_exact_duplicate_rows", "invalid_counts", "split",
        "most_correlated_numeric_pair", "outlier_capping", "projection",
    ]}
    compact["outlier_capping"] = {
        "train_rows_affected": outlier_rows, "per_feature": cap_details
    }
    compact["projection"] = {
        "pc1_pc2_variance_pct": projection["pc1_pc2_variance_pct"],
        "top_loadings": projection["top_loadings"],
    }
    compact["pipeline_shape"] = [list(x_train_t.shape), list(x_test_t.shape)]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
