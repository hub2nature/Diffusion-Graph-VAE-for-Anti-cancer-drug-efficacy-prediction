
import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

from mlp import MLP_simple

from latent import LatentDataset

import sys

# Define log file path
log_file_path = "train.txt"

# Redirect stdout and stderr to log file
sys.stdout = open(log_file_path, "w")
sys.stderr = sys.stdout  # Redirect stderr as well


# cuda = False
# DEVICE = torch.device("cuda" if cuda else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)
learning_rate = 0.001
import torch
import numpy as np
import random

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # If using numpy
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage
set_seed(42)


# dataFile = os.path.join('/content/drive/MyDrive/VAE_geneVAE/gooddrug+goodgene.txt')

dataFile = os.path.join('final_merged_drug_gene_vectors.txt')
# Adjust the 'sep' parameter according to the delimiter used in your file (e.g., '\t' for tab, ',' for comma)
df = pd.read_csv(dataFile,sep='\t')  # Change '\t' to the appropriate delimiter
df=df.dropna()
# Print the column names
print("Columns in the data file are:")
print(df.columns.tolist())
print(df.shape)



num=0
while num<2:
    num += 1
    print("-----------------------------\n-------------"+str(num)+"---------------\n-----------------------------")

    data = pd.read_csv(open(dataFile), sep='\t')
    data = data.sample(frac=1).reset_index(drop=True)
    print("Data Shape:", data.shape)
    print("Data Columns:", data.columns)
    print("First Few Rows:\n", data.head())

    
    trainData = data.loc[: 999, :]
    trainDataset = LatentDataset(trainData, train0val1test2=0)
    validationData = data.loc[1000: 1164, :]
    validationDataset = LatentDataset(validationData, train0val1test2=1)
    testData = data.loc[1165: 1329, :]
    testDataset = LatentDataset(testData, train0val1test2=2)

    trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, drop_last=True)
    validationLoader = DataLoader(validationDataset, batch_size=165, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=165, drop_last=True)
    #print(validationDataset)
    


        
    model = MLP_simple()
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)


    epoch = 0
    bestR2 = 0
    bestLoss = 200
    bestEpoch = 0
    path = 'trainedModels'
    while epoch < 50:

        model.train()
        for batch in trainLoader:
            #print('batch')
            
            geLatentVec, dLatentVec, target = batch
            # print(f"geLatentVec shape: {geLatentVec.shape}, dLatentVec shape: {dLatentVec.shape}")

            # if geLatentVec.shape[0] != 50:
            #     continue

            if torch.cuda.is_available():
                geLatentVec = geLatentVec.cuda()
                dLatentVec = dLatentVec.cuda()
                target = target.cuda()
            else:
                geLatentVec = Variable(geLatentVec)
                dLatentVec = Variable(dLatentVec)
                target = Variable(target)
            out = model(geLatentVec, dLatentVec)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch += 1
        
        if epoch % 2 == 0:
            print(epoch)
            model.eval()

            for batch in validationLoader:
                geLatentVec, dLatentVec, target = batch
                if torch.cuda.is_available():
                    geLatentVec = geLatentVec.cuda()
                    dLatentVec = dLatentVec.cuda()
                    target = target.cuda()

                out = model(geLatentVec, dLatentVec)
                out = out.data.cpu().numpy().tolist()
                target = target.cpu().numpy().tolist()
                r2 = r2_score(target, out)
                
                rmse = mean_squared_error(target, out)**0.5
                # SS_tot = torch.std(target)
                # SS_res = evalLoss

                print('epoch: {}, Validation Loss: {:.6f}, R2_Score: {:.6f}'.format(epoch, rmse, r2))
                if (r2 > bestR2 and epoch > 20):
                    bestLoss = rmse
                    bestR2 = r2
                    bestEpoch = epoch
                    torch.save(model.state_dict(), path + 'modelParameters.pt')
                    print("Got a better model!")
                # print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))


                # Draw a scatter chart of true drug response and predicted drug response
                with torch.no_grad():
                    plt.scatter(target, out, c='b')
                    plt.xlabel('True drug response (IC50)', color='k')
                    plt.ylabel('Predicted drug response (IC50)', color='k')
                    plt.title("R2_score: {0: .6f}".format(r2))
                    plt.show()
                    if r2>0.81:
                        pass


        pass


    path = 'trainedModels'
    #model.load_state_dict(torch.load(os.path.join(path, 'modelParameters.pt'), map_location=torch.device('cpu')))

    model.load_state_dict(torch.load(path + 'modelParameters.pt'))
    print('\nNow testing the best model on test dataset\n')
    model.eval()
    test_r2_scores = []
    for batch in testLoader:
        geLatentVec, dLatentVec, target = batch
        geLatentVec=geLatentVec.to(DEVICE)
        dLatentVec=dLatentVec.to(DEVICE)
        target=target.to(DEVICE)
        if torch.cuda.is_available():
            geLatentVec = geLatentVec.cuda()
            dLatentVec = dLatentVec.cuda()
            target = target.cuda()

        out = model(geLatentVec, dLatentVec)

        out = out.data.cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        r2 = r2_score(target, out)
        rmse = mean_squared_error(target, out) ** 0.5
        test_r2_scores.append(r2)

        print('epoch: {}, Validation Loss: {:.6f}, R2_Score: {:.6f}'.format(bestEpoch, bestLoss, bestR2))
        print('Test Loss: {:.6f}, R2_Score: {:.6f}'.format(rmse, r2))


        #df = pd.read_csv('/content/drive/MyDrive/VAE_geneVAE/results/R2_Score(pan+cgc+eliminated+unsampledGene+unsampledDrug).txt', sep='\t')
        df = pd.DataFrame()
        # df = df.append({'id': int(len(df)),
        #                 'R2_test': r2,
        #                 'RMSE_test': rmse,
        #                 'R2_val': bestR2,
        #                 'RMSE_val': bestLoss,
        #                 'epoch': bestEpoch},
        #                 ignore_index=True)
        import pandas as pd

        # Assuming df is your DataFrame and it already exists
        new_data = pd.DataFrame({
            'id': [int(len(df))],  # Make sure the values are in a list to align with DataFrame structure
            'R2_test': [r2],
            'RMSE_test': [rmse],
            'R2_val': [bestR2],
            'RMSE_val': [bestLoss],
            'epoch': [bestEpoch]
        })

        df = pd.concat([df, new_data], ignore_index=True)

        # df.to_csv('/content/drive/MyDrive/VAE_geneVAE/results/R2_Score(pan+cgc+eliminated+unsampledGene+unsampledDrug).txt', sep='\t', index=False)
    max_test_r2 = max(test_r2_scores)
    average_test_r2 = np.mean(test_r2_scores)
    print(f'Average Test R2 Score: {average_test_r2:.6f}')
    print(f'Maximum Test R2 Score: {max_test_r2:.6f}')
#Set seed for reproducibility
# set_seed(42)

# # Load data
# dataFile = os.path.join('/content/drive/MyDrive/VAE_geneVAE/cell+drug_vec+gene_vec_diff2GCNatt_rich_adjusted.txt')
# df = pd.read_csv(dataFile, sep='\t')
# df = df.dropna()

# # Prepare datasets
# data = df.sample(frac=1).reset_index(drop=True)
# trainData = data.loc[: 999, :]
# trainDataset = LatentDataset(trainData, train0val1test2=0)
# validationData = data.loc[1000: 1164, :]
# validationDataset = LatentDataset(validationData, train0val1test2=1)
# testData = data.loc[1165: 1329, :]
# testDataset = LatentDataset(testData, train0val1test2=2)

# # Prepare data loaders
# trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True, drop_last=True)
# validationLoader = DataLoader(validationDataset, batch_size=165, drop_last=True)
# testLoader = DataLoader(testDataset, batch_size=165, drop_last=True)

# # Model setup
# model = MLP_simple().to(DEVICE)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

# # Training loop
# epoch = 0
# bestR2 = 0
# bestLoss = 200
# bestEpoch = 0
# path = 'trainedModels'

# while epoch < 50:
#     validation_r2_scores = []
#     model.train()
#     for batch in trainLoader:
#         geLatentVec, dLatentVec, target = batch
#         #geLatentVec, dLatentVec, target = batch.to(DEVICE)
#         geLatentVec = geLatentVec.to(DEVICE)
#         dLatentVec = dLatentVec.to(DEVICE)
#         target = target.to(DEVICE)
#         out = model(geLatentVec, dLatentVec)
#         loss = criterion(out, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Validation
#     if epoch % 2 == 0:
#         model.eval()
#         batch_r2_scores = [] 
#         for batch in validationLoader:

#             geLatentVec, dLatentVec, target = batch
#             #geLatentVec, dLatentVec, target = batch.to(DEVICE)
#             geLatentVec = geLatentVec.to(DEVICE)
#             dLatentVec = dLatentVec.to(DEVICE)
#             target = target.to(DEVICE)

#             out = model(geLatentVec, dLatentVec)
#             r2 = r2_score(target.cpu().numpy(), out.data.cpu().numpy())
#             rmse = mean_squared_error(target.cpu().numpy(), out.data.cpu().numpy()) ** 0.5
#             batch_r2_scores.append(r2)
#             print('epoch: {}, Validation Loss: {:.6f}, R2_Score: {:.6f}'.format(epoch, rmse, r2))
#     average_r2 = np.mean(batch_r2_scores)
#     validation_r2_scores.append(average_r2)
#     epoch += 1
# # Calculate overall average R2 for the validation phases
# overall_average_validation_r2 = np.mean(validation_r2_scores)
# print(f'Overall Average Validation R2: {overall_average_validation_r2:.6f}')



# test_r2_scores = []

# # Load the best model
# model.load_state_dict(torch.load(os.path.join(path, 'modelParameters.pt')))
# model.eval()

# # Testing loop
# for batch in testLoader:
#     geLatentVec, dLatentVec, target = batch
#     geLatentVec = geLatentVec.to(DEVICE)
#     dLatentVec = dLatentVec.to(DEVICE)
#     target = target.to(DEVICE)

#     out = model(geLatentVec, dLatentVec)
#     r2 = r2_score(target.cpu().numpy(), out.data.cpu().numpy())
#     test_r2_scores.append(r2)  # Collect R2 score for the batch

# # Calculate average R2 for testing
# average_test_r2 = np.mean(test_r2_scores)
# print(f'Average Test R2 Score: {average_test_r2:.6f}')


# Reset stdout and stderr after writing is complete
sys.stdout.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print(f"Log saved in '{log_file_path}'")





