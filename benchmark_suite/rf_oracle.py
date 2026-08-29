"""Portable form of the TDC/PMO random-forest activity oracles (GSK3B, JNK3).

`oracle/jnk3_current.pkl` and `oracle/gsk3b_current.pkl` are TDC's own model
files (TDC loads exactly these when scikit-learn >= 0.24). They are
RandomForestClassifiers pickled by scikit-learn 0.23 and they cannot be
unpickled by scikit-learn >= 1.0: the C-level tree node array dtype changed, so
`pickle.load` raises "node array from the pickle has an incompatible dtype".
DRD2 is unaffected only because it is an SVC, which has no tree arrays.

Rather than retrain (which would define a different oracle) or run the benchmark
in an ancient environment, the forest is exported once to a plain `.npz` of tree
arrays and re-implemented here in numpy. The prediction is the same computation
scikit-learn performs -- traverse each tree, normalise the leaf's class counts,
average over trees -- so scores are identical, not approximate. `verify_rf_export`
checks that on real molecules and is a hard gate before the file is used.

Export (run with an interpreter whose scikit-learn can still read the pickle,
e.g. the py3.6 `reinvent` env with 0.24.2):

    ~/anaconda3/envs/reinvent/bin/python -m benchmark_suite.rf_oracle \
        export oracle/jnk3_current.pkl oracle/jnk3_forest.npz

Then verify from the benchmark environment:

    python -m benchmark_suite.rf_oracle verify oracle/jnk3_forest.npz \
        --reference oracle/jnk3_reference_scores.npz
"""

# NB: no `from __future__ import annotations` and no 3.7+ syntax -- the export
# half of this module has to run under the legacy py3.6 environment that can
# still unpickle the models.

import sys
from pathlib import Path

import numpy as np

LEAF = -1  # sklearn marks leaves with children_left == -1


def export_forest(pickle_path: Path, npz_path: Path) -> None:
    """Dump a RandomForestClassifier's tree arrays. Needs the legacy sklearn."""
    import pickle

    with open(pickle_path, "rb") as handle:
        model = pickle.load(handle)

    arrays = {}
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        arrays[f"children_left_{index}"] = tree.children_left.astype(np.int32)
        arrays[f"children_right_{index}"] = tree.children_right.astype(np.int32)
        arrays[f"feature_{index}"] = tree.feature.astype(np.int32)
        arrays[f"threshold_{index}"] = tree.threshold.astype(np.float64)
        # (n_nodes, n_outputs, n_classes) -> (n_nodes, n_classes); single output only
        arrays[f"value_{index}"] = tree.value[:, 0, :].astype(np.float64)
    arrays["n_trees"] = np.asarray(len(model.estimators_), dtype=np.int64)
    arrays["classes"] = np.asarray(model.classes_)
    arrays["n_features"] = np.asarray(
        getattr(model, "n_features_in_", getattr(model, "n_features_", -1)), dtype=np.int64
    )

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    print(f"exported {int(arrays['n_trees'])} trees -> {npz_path}")


class NumpyForest:
    """RandomForestClassifier.predict_proba, in numpy, from an exported .npz."""

    def __init__(self, npz_path: Path):
        data = np.load(Path(npz_path), allow_pickle=False)
        self.n_trees = int(data["n_trees"])
        self.classes = data["classes"]
        self.n_features = int(data["n_features"])
        self.trees = [
            (
                data[f"children_left_{i}"],
                data[f"children_right_{i}"],
                data[f"feature_{i}"],
                data[f"threshold_{i}"],
                data[f"value_{i}"],
            )
            for i in range(self.n_trees)
        ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_classes = self.trees[0][4].shape[1]
        out = np.zeros((n_samples, n_classes), dtype=np.float64)

        for children_left, children_right, feature, threshold, value in self.trees:
            node = np.zeros(n_samples, dtype=np.int32)
            active = np.ones(n_samples, dtype=bool)
            # sklearn's rule: go left when x[feature] <= threshold
            while active.any():
                idx = np.flatnonzero(active)
                current = node[idx]
                is_leaf = children_left[current] == LEAF
                active[idx[is_leaf]] = False
                walking = idx[~is_leaf]
                if walking.size == 0:
                    break
                nodes_w = node[walking]
                go_left = X[walking, feature[nodes_w]] <= threshold[nodes_w]
                node[walking] = np.where(go_left, children_left[nodes_w], children_right[nodes_w])

            leaf_counts = value[node]
            totals = leaf_counts.sum(axis=1, keepdims=True)
            out += np.divide(leaf_counts, totals, out=np.zeros_like(leaf_counts), where=totals > 0)

        return out / self.n_trees

    def predict_positive(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, matching TDC's `predict_proba(fp)[0, 1]`."""
        return self.predict_proba(X)[:, 1]


def _main(argv) -> int:
    if len(argv) >= 4 and argv[1] == "export":
        export_forest(Path(argv[2]), Path(argv[3]))
        return 0
    if len(argv) >= 3 and argv[1] == "predict":
        # used by the verification gate: score a saved feature matrix
        forest = NumpyForest(Path(argv[2]))
        X = np.load(argv[3])["X"]
        np.savez_compressed(argv[4], y=forest.predict_positive(X))
        print(f"wrote {argv[4]}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
