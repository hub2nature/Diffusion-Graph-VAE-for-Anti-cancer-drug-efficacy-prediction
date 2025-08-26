import pandas as pd
import sys

# Define log file path
log_file_path = "merge_gene_drug.txt"

# Redirect stdout and stderr to log file
sys.stdout = open(log_file_path, "w")
sys.stderr = sys.stdout  # Redirect stderr as well

# ---------------------- Load Data Files ----------------------
no_duplicate_drug_cellline = "CellLine_Smiles_IC50(breast)_noduplicate_no_duplicates.csv"
cell_line_file = "CellLine_Smiles_IC50(breast).txt"
drug_latent_vectors_file = "yesatt_2diff_2ho_nomask_L1_rich_nodupp.csv"
gene_latent_vectors_file = "final_latent_vectors_breast.txt"

#########################################################################


##########-----------print the descriptions of the files-----------############
def load_and_print_info(file_path, sep=","):
    """Loads a dataset and prints its shape and column names."""
    try:
        df = pd.read_csv(file_path, sep=sep)
        print(f"\n===== {file_path} =====")
        print(f"Shape: {df.shape}")
        print(f"Column Names: {', '.join(df.columns)}\n")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}\n")
        return None

# Load and print dataset information
df_no_dup = load_and_print_info(no_duplicate_drug_cellline, sep=",")
df_cell_line = load_and_print_info(cell_line_file, sep="\t")  # Assuming tab-separated
df_drug_latent = load_and_print_info(drug_latent_vectors_file, sep=",")
df_gene_latent = load_and_print_info(gene_latent_vectors_file, sep="\t")  # Assuming tab-separated

print("\n===== Dataset Loading Completed =====\n")
####################################################################################



# ---------------------- Concatenate Drug Latent Vectors with Drug ID ----------------------
if df_no_dup is not None and df_drug_latent is not None:
    # Keep only 'DRUG_ID' from the first file
    df_no_dup = df_no_dup[['DRUG_ID']].drop_duplicates()

    # Concatenate instead of merging
    concatenated_df = pd.concat([df_no_dup, df_drug_latent], axis=1)

    # Save the concatenated dataframe
    output_file = "concatenated_drug_vectors.csv"
    concatenated_df.to_csv(output_file, index=False)

    # Print the concatenated file information
    print("\n===== Concatenated Drug Latent Vectors =====")
    print(f"Concatenated File Path: {output_file}")
    print(f"Shape: {concatenated_df.shape}")
    print(f"Column Names: {', '.join(concatenated_df.columns)}\n")

print("\n===== Concatenation Completed =====\n")
#####################################################################################






# ---------------------- Merge by Replicating Drug Latent Vectors ----------------------
if df_cell_line is not None and concatenated_df is not None:
    # Drop 'SMILES_expression' column
    df_cell_line = df_cell_line.drop(columns=['SMILES_expression'], errors='ignore')

    # Merge by replicating drug latent vectors based on DRUG_ID
    merged_df = df_cell_line.merge(concatenated_df, on="DRUG_ID", how="left")

    # Save the merged dataframe
    output_file = "merged_drug_replicated_vectors.csv"
    merged_df.to_csv(output_file, index=False)

    # Print merged file information
    print("\n===== Merged Drug Latent Vectors with Cell Line Data =====")
    print(f"Merged File Path: {output_file}")
    print(f"Shape: {merged_df.shape}")
    print(f"Column Names: {', '.join(merged_df.columns)}\n")

print("\n===== Replication and Merging Completed =====\n")
################################################################################








# ---------------------- Load Data Files ----------------------
merged_drug_replicated_file = "merged_drug_replicated_vectors.csv"
gene_latent_vectors_file = "final_latent_vectors_breast.txt"

# Load and print dataset information
merged_drug_df = load_and_print_info(merged_drug_replicated_file, sep=",")
gene_latent_df = load_and_print_info(gene_latent_vectors_file, sep="\t")  # Assuming tab-separated

print("\n===== Dataset Loading Completed =====\n")

# ---------------------- Standardizing Cell Line Names ----------------------
if merged_drug_df is not None and gene_latent_df is not None:
    print("\n===== Standardizing Cell Line Names in Both DataFrames =====")

    # Ensure column name is consistent in gene latent vectors
    gene_latent_df.rename(columns={"Cell Line Name": "CELL_LINE_NAME"}, inplace=True)

    # Remove "_BREAST" suffix in gene latent vectors and strip spaces
    gene_latent_df["CELL_LINE_NAME"] = gene_latent_df["CELL_LINE_NAME"].str.replace("_BREAST", "", regex=False).str.strip()
    
    # Ensure all CELL_LINE_NAME values in both dataframes are strings and strip extra spaces
    merged_drug_df["CELL_LINE_NAME"] = merged_drug_df["CELL_LINE_NAME"].astype(str).str.strip()
    gene_latent_df["CELL_LINE_NAME"] = gene_latent_df["CELL_LINE_NAME"].astype(str).str.strip()

    print("\nUnique CELL_LINE_NAME values in merged_drug_df:\n", merged_drug_df["CELL_LINE_NAME"].unique())
    print("\nUnique CELL_LINE_NAME values in gene_latent_df:\n", gene_latent_df["CELL_LINE_NAME"].unique())

########################################################################################







# ---------------------- Merge by Replicating Gene Latent Vectors ----------------------
if merged_drug_df is not None and gene_latent_df is not None:
    print("\n===== Merging Drug and Gene Latent Vectors =====")

    # Merge by replicating gene latent vectors based on CELL_LINE_NAME
    final_merged_df = merged_drug_df.merge(gene_latent_df, on="CELL_LINE_NAME", how="left")

    # Save the final merged dataframe
    output_file = "final_merged_drug_gene_vectors.txt"
    final_merged_df.to_csv(output_file, index=False, sep='\t')

    # Print final merged file information
    print("\n===== Final Merged Drug & Gene Latent Vectors =====")
    print(f"Final Merged File Path: {output_file}")
    print(f"Shape: {final_merged_df.shape}")
    print(f"Column Names: {', '.join(final_merged_df.columns)}\n")

    # Check if any CELL_LINE_NAME values in merged_drug_df did not find a match in gene_latent_df
    missing_cell_lines = merged_drug_df[~merged_drug_df["CELL_LINE_NAME"].isin(gene_latent_df["CELL_LINE_NAME"])]
    if not missing_cell_lines.empty:
        print("\n===== Warning: Some CELL_LINE_NAME values in merged_drug_df were NOT found in gene_latent_df =====")
        print(missing_cell_lines["CELL_LINE_NAME"].unique())

print("\n===== Merging Completed Successfully =====\n")

###################################################################################
# Reset stdout and stderr after writing is complete
sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print(f"Log saved in '{log_file_path}'")
