import os
import pandas as pd
import numpy as np

base_dir = "results/"
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
    # THE ONLY READ THAT WORKS ON YOUR REAL FILES - COPY-PASTE THIS
    # --------------------------------------------------------------
    def read_ho_file(path):
        # Step 1: read everything as plain text lines
        with open(path, 'r', encoding='latin1') as f:
            lines = f.readlines()

        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split on comma, keep empty fields (,, becomes ['', ''])
            fields = [x.strip() for x in line.split(',')]

            # Keep exactly first 14 fields (extra columns are silently dropped)
            fields = fields[:14]

            # If we have less than 14 → pad with NaN (this never happens in your files,
            # but keeps the code 100% safe)
            while len(fields) < 14:
                fields.append('')

            # Replace empty strings with real NaN before conversion
            fields = [np.nan if x == '' else x for x in fields]

            data.append(fields)

        # Convert to DataFrame → now 100% safe to cast to float
        df = pd.DataFrame(data)
        df = df.astype(float)        # ← this now works every time
        print(f"{path} → {len(df)} rows, {df.shape[1]} columns loaded perfectly")
        return df

    # --------------------------------------------------------------
    train_df = read_ho_file(train_path)
    test_df  = read_ho_file(test_path)

    # Split X (first 8 columns) and y (last 6 columns)
    X_train, y_train = train_df.iloc[:, :8], train_df.iloc[:, 8:]
    X_test , y_test  = test_df.iloc[:, :8] , test_df.iloc[:, 8:]

    # Save
    X_train.to_csv(os.path.join(model_dir, "X_train.csv"), index=False, float_format='%.8f', header=False)
    y_train.to_csv(os.path.join(model_dir, "y_train.csv"), index=False, float_format='%.8f', header=False)
    X_test .to_csv(os.path.join(model_dir, "X_test.csv") , index=False, float_format='%.8f', header=False)
    y_test .to_csv(os.path.join(model_dir, "y_test.csv") , index=False, float_format='%.8f', header=False)

    print(f"Successfully processed → {model_dir}\n")