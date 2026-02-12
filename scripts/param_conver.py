import numpy as np
import pandas as pd
import argparse
import os

# -----------------------------
# Reference parameters
# -----------------------------
TISO_REF = np.array([
    150.0,      # a0
    6.0,        # b0
    116.85,     # af0
    11.83425    # bf0
])

HO8_REF = np.array([
    150.0,   # a0
    6.0,     # b0
    116.85,  # af0
    11.83425,# bf0
    372.0,   # as0
    5.16,    # bs0
    410.0,   # afs0
    11.3     # bfs0
])

PARAM_NAMES = {
    "tiso": ["a0","b0","af0","bf0"],
    "ho8":  ["a0","b0","af0","bf0","as0","bs0","afs0","bfs0"]
}

REFS = {
    "tiso": TISO_REF,
    "ho8": HO8_REF
}

# -----------------------------
# Conversion
# -----------------------------
def convert_physical_to_normalized_df(df, model="tiso"):
    refs = REFS[model]
    names = PARAM_NAMES[model]

    if df.shape[1] != len(refs):
        raise ValueError(
            f"Model {model} expects {len(refs)} parameters, "
            f"but CSV has {df.shape[1]} columns"
        )

    X = df.values.astype(float)   # (nsamples, npar)
    X_norm = X / refs[None, :]    # exact inverse mapping

    return pd.DataFrame(X_norm, columns=names)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert physical-parameter CSV to normalized space")
    parser.add_argument("--x", required=True, help="Path to X.csv (physical parameters)")
    parser.add_argument("--y", required=False, help="Path to Y.csv (outputs, optional, unchanged)")
    parser.add_argument("--out_x", required=True, help="Output path for normalized X.csv")
    parser.add_argument("--model", choices=["tiso", "ho8"], required=True, help="Model type")

    args = parser.parse_args()

    # Load X
    X_df = pd.read_csv(args.x)

    print("\nColumns:", list(X_df.columns))
    print("Shape:", X_df.shape)
    print("\nMin per column:\n", X_df.min())
    print("\nMax per column:\n", X_df.max())

    # Convert
    X_norm_df = convert_physical_to_normalized_df(X_df, model=args.model)

    # Save normalized X
    out_dir = os.path.dirname(args.out_x)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    X_norm_df.to_csv(args.out_x, index=False)
    print(f"\nNormalized X saved to: {args.out_x}")

    # Optional Y copy (unchanged)
    if args.y is not None:
        Y_df = pd.read_csv(args.y)
        out_y = os.path.join(out_dir if out_dir else ".", "Y_norm.csv")
        Y_df.to_csv(out_y, index=False)
        print(f"Y copied unchanged to: {out_y}")

    # Diagnostics
    print("\nNormalized parameter ranges:")
    for col in X_norm_df.columns:
        print(f"{col}: [{X_norm_df[col].min():.4f}, {X_norm_df[col].max():.4f}]")
