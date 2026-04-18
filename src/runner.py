import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from src.schemas import PlanSpec


def featurize_morgan(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    features = []
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append(np.zeros(n_bits, dtype=np.float32))
            continue

        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)

    return np.vstack(features)


def featurize_rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    raise NotImplementedError("RDKit descriptor featurization will be added in Version 2.")


def build_model(plan: PlanSpec):
    if plan.model_type == "ridge":
        return Ridge(alpha=plan.params.get("alpha", 1.0))

    if plan.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=plan.params.get("n_estimators", 300),
            max_depth=plan.params.get("max_depth", 12),
            random_state=plan.params.get("random_state", 42),
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_type: {plan.model_type}")


def build_features(plan: PlanSpec, train_df: pd.DataFrame, test_df: pd.DataFrame):
    if plan.feature_type == "morgan_fingerprint":
        radius = plan.params.get("radius", 2)
        n_bits = plan.params.get("n_bits", 2048)

        X_train = featurize_morgan(train_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)
        X_test = featurize_morgan(test_df["SMILES"].tolist(), radius=radius, n_bits=n_bits)
        return X_train, X_test

    if plan.feature_type == "rdkit_descriptors":
        X_train = featurize_rdkit_descriptors(train_df["SMILES"].tolist())
        X_test = featurize_rdkit_descriptors(test_df["SMILES"].tolist())
        return X_train, X_test

    raise ValueError(f"Unsupported feature_type: {plan.feature_type}")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "r2": float(r2),
    }


def save_plan_metrics(metrics: dict, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_predictions(df: pd.DataFrame, output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def run_plan(plan: PlanSpec, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train, X_test = build_features(plan, train_df, test_df)
    y_train = train_df["pEC50"].values.astype(float)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_fold_preds = []

    for train_idx, valid_idx in kf.split(X_train):
        X_tr, X_va = X_train[train_idx], X_train[valid_idx]
        y_tr = y_train[train_idx]

        model = build_model(plan)
        model.fit(X_tr, y_tr)

        oof_preds[valid_idx] = model.predict(X_va)
        test_fold_preds.append(model.predict(X_test))

    test_preds = np.mean(np.vstack(test_fold_preds), axis=0)

    metrics = compute_regression_metrics(y_train, oof_preds)
    metrics.update(
        {
            "plan_id": plan.plan_id,
            "plan_name": plan.name,
            "feature_type": plan.feature_type,
            "model_type": plan.model_type,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
        }
    )

    oof_df = pd.DataFrame(
        {
            "SMILES": train_df["SMILES"],
            "Molecule Name": train_df["Molecule Name"],
            "y_true": y_train,
            "y_pred": oof_preds,
        }
    )

    test_pred_df = pd.DataFrame(
        {
            "SMILES": test_df["SMILES"],
            "Molecule Name": test_df["Molecule Name"],
            "pEC50": test_preds,
        }
    )

    save_plan_metrics(metrics, output_path / "metrics.json")
    save_predictions(oof_df, output_path / "oof_predictions.csv")
    save_predictions(test_pred_df, output_path / "test_predictions.csv")

    return metrics
