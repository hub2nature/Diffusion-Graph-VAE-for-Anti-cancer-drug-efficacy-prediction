import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------- Redirect Print Output to Log File ----------------------
log_file_path = "vae_training_log.txt"
log_file = open(log_file_path, "w")
sys.stdout = log_file  # Redirects print statements to file
sys.stderr = log_file

# ---------------------- Load Training Data ----------------------
print("Loading Train and Validation Data...")
train_data = torch.load("train_data.pt")
val_data = torch.load("val_data.pt")

batch_size = 100
train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)

print(f"Train Data Size: {len(train_data)}, Validation Data Size: {len(val_data)}")

# ---------------------- Model Hyperparameters ----------------------
x_dim = 1001
hidden_dim = 400
latent_dim = 256
lr = 1e-3
epochs = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {DEVICE}")

# ---------------------- Define VAE Model ----------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h = self.LeakyReLU(self.batch_norm1(self.FC_input(x)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.batch_norm2(self.FC_hidden(h)))
        h = self.dropout(h)
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var

# ---------------------- Instantiate and Train VAE ----------------------
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(Encoder=encoder, Decoder=decoder).to(DEVICE)

# ---------------------- Define Loss & Optimizer ----------------------
MSE_loss = nn.MSELoss()

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = MSE_loss(x_hat, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------------------- Training VAE ----------------------
train_losses = []
val_losses = []

print("\n=================== Start Training VAE ===================\n")
for epoch in range(epochs):
    model.train()
    overall_loss = 0
    for batch in train_loader:
        x = batch[0].to(DEVICE)
        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    train_loss = overall_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation Phase
    model.eval()
    overall_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_val = batch[0].to(DEVICE)
            x_hat_val, mean_val, log_var_val = model(x_val)
            val_loss = loss_function(x_val, x_hat_val, mean_val, log_var_val)
            overall_val_loss += val_loss.item()

    val_loss = overall_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("\n=================== Training Complete ===================\n")

# ---------------------- Save Model ----------------------
torch.save(model.state_dict(), "vae_model.pth")
print("Trained model saved as 'vae_model.pth'")

# ---------------------- Save Loss Plot ----------------------
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
print("Loss plot saved as 'training_validation_loss.png'")

# ---------------------- Load Processed Breast Cancer Data ----------------------
breast_data_path = "processed_breast_cell_lines.txt"
print(f"Loading processed breast cancer cell line data from {breast_data_path}")

df_breast = pd.read_csv(breast_data_path, sep="\t", index_col=0)
data_breast = df_breast.values
data_tensor = torch.tensor(data_breast, dtype=torch.float32).to(DEVICE)

# ---------------------- Generate DataLoader ----------------------
test_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=False)

# ---------------------- Load Trained VAE Model ----------------------
print('Loading trained VAE model...')

# Ensure the model is defined before loading the weights
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(DEVICE)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim).to(DEVICE)
vae = VAE(encoder, decoder).to(DEVICE)  # Ensure `vae` is instantiated
vae.load_state_dict(torch.load("vae_model.pth", map_location=DEVICE))
vae.eval()
print("VAE model successfully loaded!")

# ---------------------- Generate Latent Vectors ----------------------
print("Generating latent vectors...")
latent_vectors = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch[0].to(DEVICE)
        mean, log_var = vae.Encoder(batch)
        z = vae.reparameterization(mean, torch.exp(0.5 * log_var))
        latent_vectors.append(z.cpu().numpy())

latent_vectors = np.concatenate(latent_vectors, axis=0)
np.savetxt("latent_vectors_breast.txt", latent_vectors, fmt="%s", delimiter="\t")
print(f"Latent vectors saved with shape: {latent_vectors.shape}")
# ---------------------- Perform PCA ----------------------
print("Performing PCA on latent vectors...")
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)

plt.figure(figsize=(10, 6))
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Breast Cancer Latent Vectors")
plt.savefig("PCA_breast.png", dpi=300, bbox_inches="tight")
plt.close()

print("All tasks completed successfully!")




# ---------------------- Load `cell_lines_breast.txt` ----------------------
breast_file_path = "cell_lines_breast.txt"
df_breast = pd.read_csv(breast_file_path, sep="\t", index_col=0)  # Load file with gene_id as index

# Extract column names that contain "BREAST"
breast_cell_lines = [col for col in df_breast.columns if "BREAST" in col]
print(f"Extracted {len(breast_cell_lines)} breast cell line names.")

# ---------------------- Load Latent Vectors ----------------------
latent_vectors_path = "latent_vectors_breast.txt"
latent_vectors = np.loadtxt(latent_vectors_path, delimiter="\t")

# Ensure the number of rows in latent vectors matches the number of extracted cell lines
if len(breast_cell_lines) != latent_vectors.shape[0]:
    raise ValueError(
        f"Mismatch: {len(breast_cell_lines)} cell line names but {latent_vectors.shape[0]} latent vectors."
    )

# ---------------------- Combine Cell Line Names with Latent Vectors ----------------------
# Convert to DataFrame
df_latent = pd.DataFrame(latent_vectors)
df_latent.insert(0, "Cell_Line_Name", breast_cell_lines)  # Add cell line names as the first column

# ---------------------- Save Updated Latent Vectors ----------------------
updated_latent_vectors_path = "updated_latent_vectors_breast.txt"
df_latent.to_csv(updated_latent_vectors_path, sep="\t", index=False)
print(f"Updated latent vectors saved to {updated_latent_vectors_path}")
# ---------------------- Load the Updated Latent Vectors ----------------------
updated_latent_vectors_path = "updated_latent_vectors_breast.txt"
df_latent = pd.read_csv(updated_latent_vectors_path, sep="\t")

# ---------------------- Update Column Headers ----------------------
num_latent_features = df_latent.shape[1] - 1  # Excluding the first column (Cell Line Name)
new_headers = ["Cell Line Name"] + [f"g{i}" for i in range(1, num_latent_features + 1)]
df_latent.columns = new_headers  # Assign new headers

# ---------------------- Save the Updated File ----------------------
final_latent_vectors_path = "final_latent_vectors_breast.txt"
df_latent.to_csv(final_latent_vectors_path, sep="\t", index=False)
print(f"Final latent vectors file saved to {final_latent_vectors_path} with updated headers.")

sys.stdout = sys.__stdout__
log_file.close()
