import sys
import torch
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from data_processing import process_dataset



# Open a log file to save all outputs
log_file = open("output_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file


torch.manual_seed(42)  # Sets the seed for CPU and GPU operations

# #---------------------------------breast cancer cell line data ---------------------------------
# file_path = '/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/GVAE/TestVAE2/new_gene_file_transpose.txt'
# df = pd.read_csv(file_path,sep="\t")
# num_ge_columns = df.shape[1] - 1  # Subtracting one for the cell_line_name column

# # Create headers list: ['cell_line_name', 'ge0', 'ge1', ..., 'geN']
# headers = ['CELL_LINE_NAME'] + [f'ge{i}' for i in range(num_ge_columns)]

# # Print row indices (first column values)
# print("\nRow Names (First 10):")
# print(df.iloc[:10, 0].tolist())  # Prints first 10 row identifiers
# print("\nTotal Rows:", df.shape[0])
# print("Total Columns:", df.shape[1])





##############################################################################################



#############-----------------all cell line data---------------------################################
file_path = 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt'
df = pd.read_csv(file_path, sep='\t')  # Adjust the separator if necessary

# Count rows and columns
num_rows, num_columns = df.shape
print(df.columns)

print(f'Number of Rows of all cell lines: {num_rows}')
print(f'Number of Columns of all cell lines: {num_columns}')

column_names = df.columns.tolist()
cell_line_types = [col.split('_', 1)[1] if '_' in col else 'UNKNOWN' for col in column_names[2:]]
cell_line_counts = Counter(cell_line_types)
cell_line_df = pd.DataFrame(cell_line_counts.items(), columns=['Cell Line Type', 'Count'])
print(cell_line_df)

########################################################################################################################################





########################################################################################################################################

#------------------------create cell line type data for MLP-----------------------------------------------------------------------------
meta_cols = ['gene_id', 'transcript_ids']
cell_line_cols = df.columns[2:]

# Group columns by cell line type
cell_line_groups = defaultdict(list)
for col in cell_line_cols:
    cell_type = col.split('_', 1)[1] if '_' in col else 'UNKNOWN'
    cell_line_groups[cell_type].append(col)

# Define selected cell line types
selected_types = {"BREAST", "PROSTATE", "LUNG", "SKIN", "LIVER", "STOMACH"}

# Process and save files
for cell_type in selected_types:
    if cell_type in cell_line_groups:
        df_subset = df[meta_cols + cell_line_groups[cell_type]]
        output_path = f"cell_lines_{cell_type.lower()}.txt"
        df_subset.to_csv(output_path, sep='\t', index=False)
        print(f"Saved {output_path} | Rows: {df_subset.shape[0]}, Columns: {df_subset.shape[1]}")

###############################################################################################################################################################



###########################################################################################################################################################
##############-----------------------remove breast or any other specific cell line type from all cell line----------######################################

# # Paths to files
# dataset_path = "/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/GVAE/TestVAE2/CCLE_RNAseq_rsem_genes_tpm_20180929.txt"
# columns_to_remove_path = "/mnt/research/Datta_Aniruddha/Students/Mondal_Madhurima/GVAE/cell_lines_breast.txt"
# df = pd.read_csv(dataset_path, sep='\t')
# columns_to_remove = pd.read_csv(columns_to_remove_path, sep='\t', nrows=0).columns.tolist()

# # Drop selected columns
# df_filtered = df.drop(columns=columns_to_remove, errors='ignore')

# # Save the filtered dataset
# filtered_dataset_path = "filtered_breast_CCLE_RNAseq_rsem_genes_tpm.txt"
# df_filtered.to_csv(filtered_dataset_path, index=False, sep='\t')
# print(f"Saved: {filtered_dataset_path}")
# print(f"Rows: {df_filtered.shape[0]}, Columns: {df_filtered.shape[1]}")


###############################################################################################################################################






###################################-----------------dataset split-----------------------------######################################################################
# ---------------------- Process Datasets ----------------------
# Define file paths
all_cell_lines_path = "CCLE_RNAseq_rsem_genes_tpm_20180929.txt"
breast_cell_lines_path = "cell_lines_breast.txt"

# Define output paths
all_cell_lines_output = "processed_all_cell_lines.txt"
breast_cell_lines_output = "processed_breast_cell_lines.txt"

# Process All Cell Lines
print("\nProcessing All Cell Line Data...")
df_all, data_all = process_dataset(all_cell_lines_path, all_cell_lines_output, num_columns_to_sample=1000)
print(f"Processed All Cell Lines Shape: {df_all.shape}")
# Process Breast Cell Lines (Just Processed, No Splitting)
print("\nProcessing Breast Cancer Cell Line Data...")
df_breast, data_breast = process_dataset(breast_cell_lines_path, breast_cell_lines_output, num_columns_to_sample=1000)
print(f"Processed Breast Cell Lines Shape: {df_breast.shape}")
# ---------------------- Convert All Cell Line Data to PyTorch Tensor ----------------------
data_tensor_all = torch.tensor(data_all, dtype=torch.float32)

# ---------------------- Split Train and Validation Data ----------------------
train_data, val_data = train_test_split(data_tensor_all, test_size=0.2, random_state=42)

# Save as PyTorch tensors (without "all_cell_lines_" prefix)
torch.save(train_data, "train_data.pt")
torch.save(val_data, "val_data.pt")

print(f"\nTrain Size: {len(train_data)}, Validation Size: {len(val_data)}")
print("Saved train_data.pt and val_data.pt")

# ---------------------- Create DataLoaders ----------------------
batch_size = 100
train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

print("\nData processing, train-validation split, and DataLoader creation completed!")

# Close log file
sys.stdout = sys.__stdout__
log_file.close()