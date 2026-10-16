"""Generate the two figures of the paper.

  paper/figures/class_dist.pdf       training records per specialty, with and without keywords
  paper/figures/confusion_model3.pdf confusion matrix of Clinical BERT + duplicate-note rule

Inputs: train_clean.csv (Data_Profile.ipynb) and
        figures/confusion_model3_counts.csv (Model3_ClinicalBERT.ipynb, section 8.5).
Usage:  python paper_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHORT = {"Cardiovascular-Pulmonary": "Cardio", "Dermatology": "Derm", "Gastroenterology": "Gastro",
         "General Medicine": "GenMed", "Neurology": "Neuro", "Neurosurgery": "NeuroSurg",
         "Obstetrics-Gynecology": "ObGyn", "Ophthalmology": "Ophth", "Orthopedic": "Ortho",
         "Psychiatry-Psychology": "Psych", "Radiology": "Radio", "Surgery": "Surgery"}
OUT = Path("paper/figures")


def class_distribution():
    train = pd.read_csv("train_clean.csv", sep=";", dtype=str, keep_default_na=False)
    n = train["medical_specialty"].value_counts()
    empty = (train[train["keywords"].str.strip() == ""]["medical_specialty"]
             .value_counts().reindex(n.index).fillna(0))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(n))
    ax.bar(x, n - empty, color="#4C72B0", label="keywords present")
    ax.bar(x, empty, bottom=n - empty, color="#DD8452", label="keywords empty")
    for i, v in enumerate(n):
        ax.text(i, v + 8, str(v), ha="center", fontsize=5.5)
    ax.set_xticks(x, [SHORT[l] for l in n.index], rotation=60, fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("training records", fontsize=7)
    ax.legend(fontsize=6, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "class_dist.pdf")


def confusion():
    cm = pd.read_csv("figures/confusion_model3_counts.csv", index_col=0)
    counts = cm.to_numpy()
    norm = counts / counts.sum(axis=1, keepdims=True)
    labels = [SHORT[l] for l in cm.index]
    fig, ax = plt.subplots(figsize=(3.4, 3.15))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=5.5,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=60, fontsize=6, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_ylabel("gold", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "confusion_model3.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    class_distribution()
    confusion()
    print("written:", *sorted(p.name for p in OUT.glob("*.pdf")))
