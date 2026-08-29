"""
VideoRAG Evaluation Harness — shared config, plotting style, and video/case
ID registry for all eval_*.py scripts in this directory.

Run scripts from the VideoRAG repo root with the project venv active:
    source .venv/bin/activate
    python evaluation/scripts/eval_01_timestamp_fidelity.py
"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

# PAPER_MODE=1 renders title-free figures (the paper supplies its own
# captions) into paper/figures/ instead of evaluation/figures/, leaving the
# titled standalone-report figures untouched.
PAPER_MODE = os.environ.get("PAPER_MODE") == "1"
FIGURES_DIR = (REPO_ROOT / "paper" / "figures") if PAPER_MODE else (REPO_ROOT / "evaluation" / "figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def maybe_title(ax_or_fig, text: str, **kwargs):
    """Set a title/suptitle unless PAPER_MODE is on (the paper's own LaTeX
    caption carries the description instead)."""
    if PAPER_MODE:
        return
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(text, **kwargs)
    else:
        ax_or_fig.suptitle(text, **kwargs)

sys.path.insert(0, str(REPO_ROOT))

BACKEND_URL = "http://127.0.0.1:8000"

# ── Video registry (rebuilt after an environment data reset — see EVALUATION.md) ──
VIDEOS = {
    "session1": "94731c1d-387",     # synthetic witness stmt A (case_eval_synth)
    "session2": "be7b9715-a02",     # synthetic witness stmt B — contradicts A
    "session3": "e8a749f2-0bb",     # synthetic witness stmt C — consistent w/ A, contradicts B
    "obama": "9fa69b5d-e50",        # real public-domain speech, 214s
    "detection": "a457b964-073",    # synthetic detection/OCR test clip
    "clancy1": "b2a91a55-681",      # real courtroom footage, 1395s, 4 speakers
    "clancy2": "b730910d-141",      # real courtroom footage, 1316s, 1 speaker
}
CASES = {
    "synth": "case_eval_synth",
    "real": "case_eval_real",
    "clancy": "case_clancy_trial",
}

# ── Okabe-Ito colorblind-safe palette (peer-reviewed standard for scientific figures) ──
OI = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "gray": "#8C8C8C",
}
CATEGORICAL_ORDER = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"],
                      OI["sky_blue"], OI["pink"], OI["yellow"], OI["gray"]]


def apply_paper_style():
    """Matplotlib rcParams for print-ready, colorblind-safe, single-hue-per-series
    figures: thin marks, recessive gridlines, no chartjunk, one axis only."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#1A1A1A",
        "axes.labelcolor": "#1A1A1A",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
    })
    return plt


def save_metrics(name: str, data: dict):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  wrote {path}")


def savefig(fig, name: str):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    print(f"  wrote {path}")
