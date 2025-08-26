import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import sys

###################------------------output file-------------################################################
log_file_path = "smiles_training_log.txt"
log_file = open(log_file_path, "w")
sys.stdout = log_file  # Redirects print statements to file
sys.stderr = log_file

# Check if CUDA (GPU) is available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

##############################################################################################################



##################-------------------f node eature extraction of smiles strings----------------#############################
class FeatureEmbedder(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(FeatureEmbedder, self).__init__()
        # Create a linear layer for each feature
        self.embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_features)])
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply each embedding to the corresponding feature
        outputs = [self.relu(embed(x[:, i:i+1])) for i, embed in enumerate(self.embeddings)]
        # Concatenate all embeddings to form a single feature vector per atom
        return torch.cat(outputs, dim=1)

def get_atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),               # Atomic number
        atom.GetMass(),                    # Atomic mass
        atom.GetHybridization().real,      # Hybridization state, converted to a real number
        1 if atom.GetIsAromatic() else 0,  # Boolean to integer for aromaticity
        atom.GetTotalValence(),            # Total valence
        atom.GetNumExplicitHs()            # Number of explicit hydrogens
    ])


scaler = StandardScaler()
file_path = "substances.csv"

# df = pd.read_csv(file_path, header=None, names=["smiles"])  # for smiles_gdsc_train.txt
df = pd.read_csv(file_path)  

max_atoms = 96 #df['smiles'].apply(lambda x: Chem.MolFromSmiles(x).GetNumAtoms()).max()
all_features = np.vstack([get_atom_features(atom) for smiles in df['smiles'] for atom in Chem.MolFromSmiles(smiles).GetAtoms()])
scaler.fit(all_features)  # Fit the scaler on all atom features
embedder = FeatureEmbedder(6, 10)  # 6 features, each embedded into 10 dimensions
##########################################################################################################################################


#################-----------------graph construction from smiles strings-----------#################################################################
def smiles_to_graph_padded(smiles, max_atoms):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, None
    node_features = np.array([get_atom_features(atom) for atom in molecule.GetAtoms()])
    node_features = scaler.transform(node_features)  # Transform (not fit) features
    node_features = torch.tensor(node_features, dtype=torch.float)
    embedded_features = embedder(node_features)  # Embed features
    adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)
    padded_features = np.pad(embedded_features.detach().numpy(), ((0, max_atoms - len(embedded_features)), (0, 0)), 'constant')
    padded_adjacency_matrix = np.pad(adjacency_matrix, ((0, max_atoms - adjacency_matrix.shape[0]), (0, max_atoms - adjacency_matrix.shape[1])), 'constant')
    return padded_features, padded_adjacency_matrix


graph_data_list = [smiles_to_graph_padded(smiles, max_atoms) for smiles in df['smiles']]
graph_data_tensors = [(torch.tensor(features, dtype=torch.float), torch.tensor(adjacency, dtype=torch.float)) for features, adjacency in graph_data_list if features is not None]
print(f"Processed {len(graph_data_tensors)} molecules.")

###########################################################################################################################################


#################-----------------train-test-val split--------######################################################################################

features, adjacencies = zip(*graph_data_tensors)
features_tensor = torch.stack(features)
adjacencies_tensor = torch.stack(adjacencies)

# First, split the data into training and other sets (combining validation and test temporarily)
features_train, features_other, adj_train, adj_other = train_test_split(
    features_tensor, adjacencies_tensor, test_size=0.4, random_state=42)

# Now split the other set into validation and test sets
features_val, features_test, adj_val, adj_test = train_test_split(
    features_other, adj_other, test_size=0.5, random_state=42)

# Use the training features as "targets" for training, validation, and test sets
targets_train = features_train
targets_val = features_val
targets_test = features_test

# Create TensorDatasets for each set
train_dataset = TensorDataset(features_train, adj_train, targets_train)
val_dataset = TensorDataset(features_val, adj_val, targets_val)
test_dataset = TensorDataset(features_test, adj_test, targets_test)

batch_size=100
# Optionally, create DataLoaders for each dataset if needed for model training
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

print(features_tensor.shape)

############################################################################################################################################################



######################---------------DConv and DGVAE model----------------#############################################################################################
"""## With adj matrix reconstruction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionConvLayer(nn.Module):
    def __init__(self, in_features, out_features, diffusion_steps, use_attention=False, higher_order=2):
        super(DiffusionConvLayer, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.use_attention = use_attention
        self.higher_order = higher_order
        self.linear = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)
        if self.use_attention:
            self.attention_weights = nn.Parameter(torch.Tensor(1, in_features))
            nn.init.xavier_uniform_(self.attention_weights, gain=1.414)
        self.norm = nn.LayerNorm(out_features)

    # def forward(self, adjacency_matrix, node_features, edge_features=None):
    #     if adjacency_matrix.size(1) != adjacency_matrix.size(2):
    #         raise ValueError("Adjacency matrix must be square (N x N).")

    #     deg = adjacency_matrix.sum(1)
    #     deg_inv_sqrt = torch.pow(deg, -0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)

    #     if edge_features is not None:
    #         adjacency_matrix = adjacency_matrix * edge_features

    #     norm_adj = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

    #     adj_power = norm_adj
    #     for i in range(1, self.higher_order):
    #         adj_power = adj_power @ norm_adj

    #     x = node_features
    #     for _ in range(self.diffusion_steps):
    #         if self.use_attention:
    #             attention_score = F.softmax(self.attention_weights.matmul(x.transpose(0, 1)), dim=1)
    #             x = adj_power @ (x * attention_score)
    #         else:
    #             x = adj_power @ x
    #         x = F.relu(self.linear(x))
    #         x = self.norm(x)

    #     return x

    def forward(self, adjacency_matrix, node_features, edge_features=None):
      #print("Initial node features shape:", node_features.shape)
      if adjacency_matrix.size(1) != adjacency_matrix.size(2):
          raise ValueError("Adjacency matrix must be square (N x N).")

      deg = adjacency_matrix.sum(1)
      deg_inv_sqrt = torch.pow(deg, -0.5)
      deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
      #mask = adjacency_matrix != 0

      norm_adj = D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
      #norm_adj = D_inv_sqrt @ (adjacency_matrix * mask.float()) @ D_inv_sqrt
      #adj_power = norm_adj
      adj_power = norm_adj

      for i in range(1, self.higher_order):
          adj_power = adj_power @ norm_adj

      x = node_features
      for _ in range(self.diffusion_steps):
          x = adj_power @ x
          #print("Shape after diffusion:", x.shape)

      x = F.relu(self.linear(x))
      #print("Shape after linear transformation:", x.shape)

      return x

############TO Do #########################
# add trailabanle parametrs - add them both before /after diffusion steps - add resnet kind of things 
class DiffusionConvNetwork(nn.Module):
    def __init__(self, features_list, diffusion_steps):
        super(DiffusionConvNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(features_list)):
            self.layers.append(DiffusionConvLayer(features_list[i-1], features_list[i], diffusion_steps))

    def forward(self, adjacency_matrix, node_features):
        x = node_features
        #print("Input size:", node_features.size())
        for layer in self.layers:
            x = layer(adjacency_matrix, x)
            #print("After linear size:", x.size())
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
def convert_probabilities_to_binary(matrix, threshold=0.55):
      return (matrix > threshold).float()

import torch
import torch.nn as nn
import torch.nn.functional as F
def convert_probabilities_to_binary(matrix, threshold=0.55):
      return (matrix > threshold).float()


######################################TO DO ################
#clear why mean and var is a 2d vector - or shift them into a 1d vector - check if 1d mean / var actually changes accuracy or not
class VGAE(nn.Module):
    def __init__(self, features_list, max_atoms, diffusion_steps):
        super(VGAE, self).__init__()
        self.max_atoms = max_atoms  # Ensure this is set before any methods that use it
        self.encoder = DiffusionConvNetwork(features_list[:-1], diffusion_steps)
        self.encoder_mean = DiffusionConvNetwork([features_list[-2], features_list[-1]], diffusion_steps)
        self.encoder_logvar = DiffusionConvNetwork([features_list[-2], features_list[-1]], diffusion_steps)

        out_features = features_list[-1]
        in_features = features_list[0]
        hidden_features = features_list[-2]

        # Node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(out_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, in_features),
            nn.Sigmoid()
        )

        # Adjusted adjacency decoder
        self.adj_decoder = nn.Sequential(
            nn.Linear(out_features, hidden_features),
            nn.LeakyReLU(),
            nn.Linear(hidden_features, self.max_atoms ),
            nn.Sigmoid()
        )


    def encode(self, adjacency_matrix, node_features):
        hidden = self.encoder(adjacency_matrix, node_features)
        mu = self.encoder_mean(adjacency_matrix, hidden)
        logvar = self.encoder_logvar(adjacency_matrix, hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        reconstructed_node_features = self.node_decoder(z)
        pre_reshape_adj = self.adj_decoder(z)
        #print('pre_reshape_adj',pre_reshape_adj.shape)
        reconstructed_adjacency_matrix = pre_reshape_adj.view(z.size(0), self.max_atoms, self.max_atoms)
        return reconstructed_node_features, reconstructed_adjacency_matrix
    # def decode(self, z):
    #     reconstructed_node_features = self.node_decoder(z)
    #     pre_reshape_adj = self.adj_decoder(z)
    #     # Reshape to adjacency matrix shape
    #     reconstructed_adjacency_matrix = pre_reshape_adj.view(z.size(0), self.max_atoms, self.max_atoms)
    #     # Convert probabilities to binary values if needed inside the model
    #     reconstructed_adjacency_matrix = convert_probabilities_to_binary(reconstructed_adjacency_matrix)
    #     return reconstructed_node_features, reconstructed_adjacency_matrix


    def forward(self, adjacency_matrix, node_features):
        mu, logvar = self.encode(adjacency_matrix, node_features)
        z = self.reparameterize(mu, logvar)
        reconstructed_node_features, reconstructed_adjacency_matrix = self.decode(z)
        #print('reconstructed_adjacency_matrix',reconstructed_adjacency_matrix)
        return reconstructed_node_features, reconstructed_adjacency_matrix, mu, logvar

# Assuming device is already defined
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VGAE(features_list=[60, 128,  32], max_atoms=96, diffusion_steps=2).to(device)

#################################################################################################################################


###########------------------loss function--------------------------------------###################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
import torch
import torch.nn.functional as F

def loss_function(reconstructed_node_features, node_features, reconstructed_adjacency_matrix, adjacency_matrix, mu, logvar):
    # Reconstruction loss for node features using Mean Squared Error
    node_reconstruction_loss = F.mse_loss(reconstructed_node_features, node_features)

    # Reconstruction loss for the adjacency matrix using Mean Squared Error
    adjacency_reconstruction_loss = F.mse_loss(reconstructed_adjacency_matrix, adjacency_matrix)

    # KL divergence loss
    kl_divergence_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return node_reconstruction_loss + adjacency_reconstruction_loss + kl_divergence_loss

######################################################################################################################################




#############-------------------------train and validation loop-------------###################################################################
torch.manual_seed(42)
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for features, adjacencies, _ in train_loader:
        optimizer.zero_grad()
        features = features.to(device)
        adjacencies = adjacencies.to(device)
        # print("Features shape:", features.shape)  # Expected shape: [batch_size, num_features, feature_dim]
        # print("Adjacencies shape:", adjacencies.shape)

        recon_node_features, recon_adj_matrix, mu, logvar = model(adjacencies, features)
        # print("Reconstructed Adj Matrix shape:", recon_adj_matrix.shape)
        # print("Original Adj Matrix shape:", adjacencies.shape)
        loss = loss_function(recon_node_features, features, recon_adj_matrix, adjacencies, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_losses.append(train_loss / len(train_loader.dataset))

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, adjacencies, _ in val_loader:
            features = features.to(device)
            adjacencies = adjacencies.to(device)

            recon_node_features, recon_adj_matrix, mu, logvar = model(adjacencies, features)

            loss = loss_function(recon_node_features, features, recon_adj_matrix, adjacencies, mu, logvar)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader.dataset))

    print(f'Epoch: {epoch}, Training Loss: {train_loss / len(train_loader.dataset)}, Validation Loss: {val_loss / len(val_loader.dataset)}')



import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Save the figure as an image file (e.g., PNG, JPG, PDF)
plt.savefig("training_validation_loss.png", dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close()

print("Training loss plot saved as 'training_validation_loss.png'")

#############################################################################################################################################################



####################----------------extra case studies-------------------------#############################################################################
import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

def normalize_and_threshold(matrix, method='min-max', threshold_type='median'):
    """ Normalize and threshold the matrix. """
    if method == 'min-max':
        normalized_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    elif method == 'z-score':
        normalized_matrix = (matrix - np.mean(matrix)) / np.std(matrix)

    if threshold_type == 'median':
        threshold = np.median(normalized_matrix)
    elif threshold_type == 'mean':
        threshold = np.mean(normalized_matrix)
    else:
        threshold = 0.5  # default threshold
    print(threshold)
    binary_matrix = (normalized_matrix > threshold).astype(int)
    return binary_matrix

def calculate_similarity(original_nodes, recon_nodes, original_adj, recon_adj, method='min-max', threshold_type='median'):
    """ Calculate the Jaccard similarity for adjacency matrices and cosine similarity for node features. """
    # Normalize and binarize adjacency matrices
    original_adj_bin = normalize_and_threshold(original_adj, method, threshold_type)
    recon_adj_bin = normalize_and_threshold(recon_adj, method, threshold_type)

    # Flatten the adjacency matrices for similarity calculation
    original_adj_flat = original_adj_bin.flatten()
    recon_adj_flat = recon_adj_bin.flatten()

    # Calculate Jaccard similarity for adjacency matrices
    adjacency_similarity = jaccard_score(original_adj_flat, recon_adj_flat, average='binary')

    # Calculate cosine similarity for node features
    node_similarity = 1 - cosine(original_nodes.flatten(), recon_nodes.flatten())

    return node_similarity, adjacency_similarity

def evaluate_test_set(model, test_loader, device, method='min-max', threshold_type='median'):
    """ Evaluate the test dataset using the model and calculate average similarities. """
    model.eval()  # Set the model to evaluation mode
    node_similarities = []
    adjacency_similarities = []

    with torch.no_grad():
        for features, adjacencies, _ in test_loader:
            features = features.to(device)
            adjacencies = adjacencies.to(device)

            recon_node_features, recon_adj_matrix, mu, logvar = model(adjacencies, features)

            # Convert tensors to numpy for similarity calculation
            features_np = features.cpu().numpy()
            adjacencies_np = adjacencies.cpu().numpy()
            recon_node_features_np = recon_node_features.cpu().numpy()
            recon_adj_matrix_np = recon_adj_matrix.cpu().numpy()

            # Calculate similarities using the specified normalization and thresholding methods
            node_similarity, adjacency_similarity = calculate_similarity(
                features_np, recon_node_features_np, adjacencies_np, recon_adj_matrix_np, method, threshold_type)
            node_similarities.append(node_similarity)
            adjacency_similarities.append(adjacency_similarity)

    # Calculate average similarities
    average_node_similarity = np.mean(node_similarities)
    average_adjacency_similarity = np.mean(adjacency_similarities)

    return average_node_similarity, average_adjacency_similarity

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
average_node_similarity, average_adjacency_similarity = evaluate_test_set(model, test_loader, device, 'min-max', 'median')
print(f"Average Node Feature Similarity: {average_node_similarity}")
print(f"Average Adjacency Matrix Similarity: {average_adjacency_similarity}")

##################################################################################################################




############------------generate latent vectors from separate data---------------############################################

import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from sklearn.preprocessing import StandardScaler

# Load the new dataset
df_new = pd.read_csv('CellLine_Smiles_IC50(breast)_noduplicate_no_duplicates.csv')

# Initialize the scaler and feature embedding model as before
scaler = StandardScaler()
embedder = FeatureEmbedder(6, 10)  # Assuming the feature dimensionality and embedding dimension are the same

# Calculate maximum number of atoms in the new dataset for padding
max_atoms_new = 96 #max([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in df_new['SMILES_expression'] if Chem.MolFromSmiles(smiles)])

# Gather all atom features from the new dataset to fit the scaler
all_features_new = np.vstack([get_atom_features(atom) for smiles in df_new['SMILES_expression'] for atom in Chem.MolFromSmiles(smiles).GetAtoms()])
scaler.fit(all_features_new)  # Fit the scaler on all atom features from the new dataset

# Process each molecule in the new dataset
graph_data_list_new = [smiles_to_graph_padded(smiles, max_atoms_new) for smiles in df_new['SMILES_expression']]
graph_data_tensors_new = [(torch.tensor(features, dtype=torch.float), torch.tensor(adjacency, dtype=torch.float)) for features, adjacency in graph_data_list_new if features is not None]

# Output the number of processed molecules
print(f"Processed {len(graph_data_tensors_new)} molecules from the new dataset.")


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Split the graph data tensors into features (node features) and targets (for simplicity, using features as targets here)
features_new, adjacencies_new = zip(*graph_data_tensors_new)

# Convert lists of tensors to a single tensor for features and adjacencies
features_tensor_new = torch.stack(features_new)
adjacencies_tensor_new = torch.stack(adjacencies_new)

targets_test = features_tensor_new


test_dataset = TensorDataset(features_tensor_new, adjacencies_tensor_new )
print(len(test_dataset))


batch_size = 10  # Example batch size, adjust as needed

# Create data loaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch
import numpy as np
import pandas as pd

model.eval()  # Ensure the model is in evaluation mode
latent_vectors = []

with torch.no_grad():  # No need to track gradients for inference
    for node_features, adjacency_matrix in test_loader:
        node_features = node_features.to(device)
        adjacency_matrix = adjacency_matrix.to(device)
        #print("Original node_features shape:", node_features.shape)  # Debug print

        # Calculate expected size for reshaping based on actual data
        num_samples, num_nodes, num_features = node_features.shape
        expected_size = num_samples * num_nodes * num_features  # Recalculate based on actual data

        actual_size = node_features.numel()  # Number of elements in the tensor
        #print("Expected size for reshaping based on actual data:", expected_size)
        #print("Actual size of node_features:", actual_size)

        if actual_size == expected_size:
            try:
                # Encode to get mu and logvar
                mu, logvar = model.encode(adjacency_matrix, node_features.view(num_samples, num_nodes, num_features))
                z = model.reparameterize(mu, logvar)
                z = z.mean(dim=1) * 1000
                latent_vectors.append(z.cpu().numpy())  # Assuming use of GPU
            except RuntimeError as e:
                print("Error during model encoding or reparametrization:", e)
        else:
            print(f"Cannot reshape node_features to [{num_samples}, {num_nodes}, {num_features}] due to incorrect total elements.")

# Concatenate latent vectors for further processing
if latent_vectors:
    latent_vectors_concatenated = np.concatenate(latent_vectors, axis=0)
    print("Concatenated shape:", latent_vectors_concatenated.shape)
else:
    print("No latent vectors were processed.")
import numpy as np
import pandas as pd

# First, concatenate your list of numpy arrays into a single numpy array
latent_vectors_concatenated = np.concatenate(latent_vectors, axis=0)

# Now, check if you need to flatten or reshape the concatenated array
# The error suggested you have a 3D array and you want to flatten the last two dimensions
latent_vectors_flattened = latent_vectors_concatenated.reshape(-1, latent_vectors_concatenated.shape[-1])

###########TO DO#####################
#TRY NOT FLATTENING IT AND PASS 
######################################################################################################################

############-------------------latent vectors saved------------#########################################
# Convert the flattened latent vectors to a Pandas DataFrame
latent_vectors_df = pd.DataFrame(latent_vectors_flattened)

# Generate column names based on the latent vector size
column_names = [f'd{i}' for i in range(latent_vectors_flattened.shape[1])]
latent_vectors_df.columns = column_names

# Save the DataFrame to a CSV file, including the header
csv_file_path = 'yesatt_2diff_2ho_nomask_L1_rich_nodupp.csv'
latent_vectors_df.to_csv(csv_file_path, index=False)
print(f"Number of rows in the DataFrame: {latent_vectors_df.shape[0]}")


#############################################################################################################
log_file.close()