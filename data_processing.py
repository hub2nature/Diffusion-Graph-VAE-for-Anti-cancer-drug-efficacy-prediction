import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_dataset(input_file, output_file, num_columns_to_sample=2000):
    """
    Processes a gene expression dataset:
    - Drops metadata columns
    - Transposes data
    - Samples `num_columns_to_sample` gene expression columns
    - Standardizes data
    - Saves processed dataset to `output_file`
    
    Returns:
    - df_scaled (Processed DataFrame)
    - scaled_data (NumPy array for model training)
    """

    print(f"Processing dataset: {input_file}")
    df = pd.read_csv(input_file, sep="\t")

    # Drop metadata columns
    meta_cols = ['gene_id', 'transcript_ids']
    df_filtered = df.drop(columns=meta_cols, errors="ignore")

    # Transpose DataFrame (cell lines as rows, genes as columns)
    df_transposed = df_filtered.T
    print(f"After Transposing: Rows={df_transposed.shape[0]}, Columns={df_transposed.shape[1]}")

    # Separate fixed columns and gene expression columns
    df_fixed = df_transposed.iloc[:, :2]
    df_to_sample = df_transposed.iloc[:, 2:]

    # Sample gene expression columns
    if df_to_sample.shape[1] > num_columns_to_sample:
        df_sampled = df_to_sample.sample(n=num_columns_to_sample, axis=1, random_state=42)
    else:
        print("Not enough columns to sample, keeping all.")
        df_sampled = df_to_sample

    # Combine fixed and sampled data
    df_reduced = pd.concat([df_fixed, df_sampled], axis=1)
    print(f"After Sampling: Rows={df_reduced.shape[0]}, Columns={df_reduced.shape[1]}")

    # Standardize data (excluding first column with cell line names)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_reduced.iloc[:, 1:])  # Exclude first column

    # Convert back to DataFrame
    df_scaled = pd.DataFrame(scaled_data, index=df_reduced.index, columns=df_reduced.columns[1:])
    print(f"After Scaling: Rows={df_scaled.shape[0]}, Columns={df_scaled.shape[1]}")

    # Save processed dataset
    df_scaled.to_csv(output_file, sep="\t", index=True)
    print(f"Processed dataset saved to: {output_file}")

    return df_scaled, scaled_data  # Return both DataFrame and NumPy array
