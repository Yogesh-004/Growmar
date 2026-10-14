"""Baseline candidate builders (fixed, reproducible, no tuning)."""
from __future__ import annotations

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_candidates(
    *,
    random_state: int = 42,
    logreg_solver: str = "lbfgs",
    logreg_max_iter: int = 5000,
    logreg_c: float = 1.0,
    tree_max_depth: int = 4,
) -> dict[str, object]:
    """Five prespecified candidates: dummy, logreg x2, shallow tree x2."""
    return {
        "dummy_stays": DummyClassifier(strategy="constant", constant=0),
        "logreg": LogisticRegression(
            solver=logreg_solver, max_iter=logreg_max_iter, C=logreg_c, random_state=random_state
        ),
        "logreg_balanced": LogisticRegression(
            solver=logreg_solver,
            max_iter=logreg_max_iter,
            C=logreg_c,
            class_weight="balanced",
            random_state=random_state,
        ),
        "tree": DecisionTreeClassifier(max_depth=tree_max_depth, random_state=random_state),
        "tree_balanced": DecisionTreeClassifier(
            max_depth=tree_max_depth, class_weight="balanced", random_state=random_state
        ),
    }


def clone_candidate(estimator: object) -> object:
    """Fresh unfitted copy for each fold/final fit (no state leakage)."""
    return clone(estimator)


def describe_candidates() -> list[dict[str, str]]:
    """Human-readable configuration table for reports and metadata."""
    return [
        {"name": "dummy_stays", "model": "DummyClassifier", "setting": "strategy=constant, constant=0 (always predicts stays)"},
        {"name": "logreg", "model": "LogisticRegression", "setting": "solver=lbfgs, max_iter=5000, C=1.0, unweighted"},
        {"name": "logreg_balanced", "model": "LogisticRegression", "setting": "solver=lbfgs, max_iter=5000, C=1.0, class_weight=balanced"},
        {"name": "tree", "model": "DecisionTreeClassifier", "setting": "max_depth=4, unweighted (deliberately shallow)"},
        {"name": "tree_balanced", "model": "DecisionTreeClassifier", "setting": "max_depth=4, class_weight=balanced (deliberately shallow)"},
    ]
