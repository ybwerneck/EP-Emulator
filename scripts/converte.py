import os
import pandas as pd
import numpy as np

base_dir = "cardiac_ho_uqsa/results/"
dest_dir = "Generated_Data_1K"
os.makedirs(dest_dir, exist_ok=True)

datasets = [
    ("output_ho_full_orig_train", "output_ho_full_orig_test"),
    ("output_ho_tiso_orig_train", "output_ho_tiso_orig_test")
]

for train_folder, test_folder in datasets:
    dataset_name = train_folder.replace("_train", "")
    model_dir = os.path.join(dest_dir, dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    train_path = os.path.join(base_dir, train_folder, "trainData.txt")
    test_path  = os.path.join(base_dir, test_folder , "testData.txt")

    # --------------------------------------------------------------
    # Robust reader
    # --------------------------------------------------------------
    def read_ho_file(path):
        with open(path, 'r', encoding='latin1') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = [x.strip() for x in line.split(',')]

            # Keep all columns dynamically
            fields = [np.nan if x == '' else x for x in fields]
            data.append(fields)

        df = pd.DataFrame(data).astype(float)
        print(f"{path} → {len(df)} rows, {df.shape[1]} columns loaded perfectly")
        return df

    # --------------------------------------------------------------
    train_df = read_ho_file(train_path)
    test_df  = read_ho_file(test_path)

    # --------------------------------------------------------------
    # Dynamic split
    # Rule: last 4 columns are always QoIs (Y)
    # --------------------------------------------------------------
    n_cols = train_df.shape[1]
    n_qoi  = 6

    X_train = train_df.iloc[:, :n_cols - n_qoi]
    y_train = train_df.iloc[:, n_cols - n_qoi:]

    X_test  = test_df.iloc[:, :n_cols - n_qoi]
    y_test  = test_df.iloc[:, n_cols - n_qoi:]

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    X_train.to_csv(os.path.join(model_dir, "X_train.csv"), index=False, float_format='%.8f', header=False)
    y_train.to_csv(os.path.join(model_dir, "y_train.csv"), index=False, float_format='%.8f', header=False)
    X_test .to_csv(os.path.join(model_dir, "X_test.csv") , index=False, float_format='%.8f', header=False)
    y_test .to_csv(os.path.join(model_dir, "y_test.csv") , index=False, float_format='%.8f', header=False)

    print(f"Successfully processed → {model_dir}")
    print(f"X dim: {X_train.shape[1]} | Y dim: {y_train.shape[1]} (QoIs = {n_qoi})\n")
