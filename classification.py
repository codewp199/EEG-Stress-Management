
"""Training and evaluation for SVM-based EEG stress classification.

- Extracts features (bandpower by default)
- Trains SVM with nested cross-validation (10x repeated StratifiedKFold)
- Saves model artifacts
"""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple
from joblib import dump

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Expect features & labels in .npy or .csv
def load_features_labels(feat_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if feat_path.endswith('.npy'):
        X = np.load(feat_path)
    else:
        X = np.loadtxt(feat_path, delimiter=',')
    if label_path.endswith('.npy'):
        y = np.load(label_path)
    else:
        y = np.loadtxt(label_path, delimiter=',').astype(int)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train SVM for EEG stress classification")
    parser.add_argument('--features', required=True, help='Path to features .npy or .csv')
    parser.add_argument('--labels', required=True, help='Path to labels .npy or .csv (0=relaxed,1=stressed)')
    parser.add_argument('--out', default='svm_model.joblib', help='Output model filename')
    args = parser.parse_args()

    X, y = load_features_labels(args.features, args.labels)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
    ])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy (10x10 CV): mean={scores.mean():.3f} ± {scores.std():.3f}")

    # Fit on full data and save
    pipe.fit(X, y)
    dump(pipe, args.out)
    print(f"Saved model to {args.out}")

if __name__ == '__main__':
    main()
