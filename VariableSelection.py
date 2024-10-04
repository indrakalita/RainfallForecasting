#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torchvision.models
import utils as ut
import models64 as models
import dataload as dl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_data = False # always false if not preparing the data and loading the already prepared data


if(save_data == True):
    GPM = '/GPM dataset path/'
    ERA = '/ERA5 dataset path/'
    variable_path = 'files.txt'
    GPM, ERA, _, date = dl.LoadData(GPM, ERA, variable_path, ERA_zoom = False, GPM_zoom=False, 
                                   GPM_desired_shape = (64, 64), ERA_desired_shape = (73, 54))
    #with open('/Path to save/ERA5_dry.pkl', 'wb') as f:
    #    pickle.dump(ERA, f)
    #with open('/Path to save/GPM_dry.pkl', 'wb') as f:
    #    pickle.dump(GPM, f)
    #with open('/Path to save/lonlat_GPM_ERA.pkl', 'wb') as f:
    #    pickle.dump(latlon, f)
    with open('/Path to save/date_wet.pkl', 'wb') as f:
        pickle.dump(date, f)
else:
    with open('/Path to load/date_all.pkl', 'rb') as f:
        date = pickle.load(f)
    with open('/Path to load/ERA5_64_64.pkl', 'rb') as f:
        ERA = pickle.load(f)
    with open('/Path to load/GPM_64_64.pkl', 'rb') as f:
        GPM = pickle.load(f)
        GPM = np.clip(GPM, 0, None)
print(GPM.shape, ERA.shape, np.min(GPM), np.max(GPM), date.shape, date)


Ndata = []
for i in range(len(ERA)):
    if(i<59):
        Ndata.append(ut.znorm(ERA[i]))
    else:
        print(f"No normalization for {i} as they are seasonal and time variables")
        Ndata.append([ERA[i],0,0])
ENorm = [item[0] for item in Ndata]
Emean = [item[1] for item in Ndata] #min val for 0-1 normalization
Estd = [item[2] for item in Ndata] #max val for 0-1 normalization
GNorm, Gmean, Gstd  = ut.norm(GPM)

def Preparedata(data,dataL,s,d,var, date):
    X,Y,Z = [],[],[]
    for i in range(0,len(data[0])-max(s,d)):
        T, L = [], []
        for j in var:
            x = data[j][i:i+s]
            T.append(x)
        X.append(T),Y.append(dataL[i+s:i+s+d]),Z.append(int(date[i+s:i+s+d]))#Y.append(data[0][i+s])
    return np.array(X), np.array(Y), np.array(Z)

sequence, days, variable = 1, 1, [num for num in range(61) if num < 2 or num > 5] #ravoid crwc300, 500, 600, 700, las empty channel
X,Y,Z = Preparedata(ENorm, GNorm, sequence, days, variable, date)
if(len(variable)>1):
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4])
    #Y = np.expand_dims(Y,axis=1) # 0 represents the precipition data as output
X = X.squeeze()
print('Dimension and values of the data:',X.shape, Y.shape, np.max(X), np.min(X),np.max(Y), np.min(Y),Z) #N,C,D,16,16
print('Visualization of example dataset')
for i in range(0,1):
    ut.visualize(Var1=X[i,0],Var2=X[i,1],Var3=X[i,2],Var4=X[i,3],Var5=X[i,4],Prec1=Y[i,0]) # For more variable add the other dimensions
###################Important#############################################
#If you have already saved the data as permutation_indices.npy then do not change anything in this cell
#This cell will make sure that every time we have the same set of data for testing
num_images = len(X)
split_ratio = [0.98, 0.01, 0.01]
num_train = int(num_images * split_ratio[0])
num_val = int(num_images * split_ratio[1])
num_test = num_images - num_train - num_val
if os.path.exists('/Path to check/permutation_indices.npy'):
    print("Permutation file exists and loading it...")
    permutation_indices = np.load('/Path to load/permutation_indices.npy')
else:
    print("Permutation file does not exist and preparing it for future reference")
    permutation_indices = np.random.permutation(num_images)
    np.save('/Path to save/permutation_indices.npy', permutation_indices)
print('Before randomness, the dates are:',Z)
X,Y,Z = X[permutation_indices],Y[permutation_indices],Z[permutation_indices]
print('Below, both the line should be the same')
print('After randomness, the dates are:',Z)
print('After randomness, the dates are: [20120917 20100422 20060824 ... 20210610 20080224 20010215]')


trainX, valX, testX = np.split(X, [num_train, num_train + num_val])
trainY, valY, testY = np.split(Y, [num_train, num_train + num_val])
trainZ, valZ, testZ = np.split(Z, [num_train, num_train + num_val])
GPM_train, GPM_val, GPM_test  = ut.CustomDataset(trainX, trainY), ut.CustomDataset(valX, valY), ut.CustomDataset(testX, testY)
print(len(GPM_train), len(GPM_val), len(GPM_test))

batch_size = 256
train_loader = DataLoader(GPM_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(GPM_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(GPM_test, batch_size=1, shuffle=False)
for batch_idx, (E,L) in enumerate(test_loader):
    print(E.shape, L.shape, testZ[batch_idx])
    break
print(len(trainZ),len(valZ),len(testZ))


def ModelSelection(select_model, fine_tune=False):
    if select_model==1:
        model = models.UNet(in_channels=X.shape[1], out_channels=64, n_class=days, kernel_size=3, padding=1, stride=1).to(device)
        print('Selected model UNet: A standard model without dropout, normalization')
    if select_model==2:
        model = models.UNET(in_channels=X.shape[1]).to(device) #Standard model
        print('Selected model UNET: A basic model')
    return model


# In[ ]:


# Masking the input based on delta either 0 or 1
def mask_input(x, delta):
    delta = delta.view(1, -1, 1, 1) 
    return x * delta
def update_selection_variables(delta, x, y, model, criterion, optimizer, n, B, sigma_sq, r):
    new_delta = delta.clone().to(x.device)
    K = delta.shape[0]
    j = np.random.randint(0,57)
    delta_j_0 = new_delta.clone()
    delta_j_0[j] = 0
    delta_j_1 = new_delta.clone() 
    delta_j_1[j] = 1

    masked_x_0 = mask_input(x, delta_j_0)
    masked_x_1 = mask_input(x, delta_j_1)

    masked_x_0, masked_x_1, y = masked_x_0.to(device), masked_x_1.to(device), y.to(device)
    ### Check NAN value only######################################################################
    if torch.isnan(masked_x_0).any() or torch.isinf(masked_x_0).any():
            print("masked_x_0 has NaN or Inf values!")
    if torch.isnan(masked_x_1).any() or torch.isinf(masked_x_1).any():
            print("masked_x_1 has NaN or Inf values!")
    for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print("Model parameters contain NaN or Inf values!")
    #################################################################################################
    with torch.no_grad():  # Avoid storing gradients
            loss_0 = criterion(model(masked_x_0), y)
            loss_1 = criterion(model(masked_x_1), y)
    #print(loss_0.shape, loss_0)
    

    avg_loss_0 = loss_0#torch.mean(loss_0)
    avg_loss_1 = loss_1#torch.mean(loss_1)

    p_j_hat = torch.sigmoid(-(r + (n / (2 * sigma_sq)) * (avg_loss_1 - avg_loss_0)))
    new_delta[j] = torch.bernoulli(p_j_hat)
        
    return new_delta, j, p_j_hat



path = '/Models/'
model_path = os.path.join(path,'Model1_L1Loss_Timezone3.pth') #Model1_L1Loss_Timezone3.pth, Model0_MSE_Timezone3.pth
print("Loading model...")
model = ModelSelection(1,fine_tune=False)
model.load_state_dict(torch.load(model_path))
x = torch.randn(1,X.shape[1], 64, 64).to(device)
pred = model(x)
print(pred.shape)


#Hyparameters
import math
from torch.optim import lr_scheduler
from torch.optim import Adam, SGD
learning_rate, num_epochs, step_size, gamma = 0.00000001, 1000, 7000, 0.1
#optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #0.00001
optim = SGD(model.parameters(), lr=learning_rate)  # Use SGD for SGLD

criterion = nn.MSELoss()
scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=gamma, patience=step_size, verbose=True)
best_train_loss, current_patience, patience = np.inf, 0, 1200

#new hyparameters
n = len(train_loader.dataset) # number of training sample
B = batch_size  # Mini-batch size
sigma_sq = 0.1
rho = 0.0001
#Selection Variables
K = 57  # Number of weather variables ERA5 55+2
r = math.log(18)  # Hyperparameter for Bernoulli prior (adjust as needed)


def StatPlot(stats, output, epoch):
    # Prepare data for plotting
    identities_list = list(stats.keys())
    frequencies = [stats[identity]['frequency'] for identity in identities_list]
    means = [stats[identity]['mean'] for identity in identities_list]
    medians = [stats[identity]['median'] for identity in identities_list]
    std_devs = [stats[identity]['std'] for identity in identities_list]
    mins = [stats[identity]['min'] for identity in identities_list]
    maxs = [stats[identity]['max'] for identity in identities_list]

    sorted_identities_by_mean = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)  #Highest to lowest
    sorted_identities_list = [identity[0] for identity in sorted_identities_by_mean]

    
    # Plot the statistics
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Statistics for Each Identity')
    
    axs[0, 0].bar(identities_list, frequencies)
    axs[0, 0].set_title('Frequency')
    axs[0, 0].set_xlabel('Identity')
    axs[0, 0].set_ylabel('Frequency')
    
    axs[0, 1].bar(identities_list, means)
    axs[0, 1].set_title('Mean')
    axs[0, 1].set_xlabel('Identity')
    axs[0, 1].set_ylabel('Mean')
    
    axs[1, 0].bar(identities_list, medians)
    axs[1, 0].set_title('Median')
    axs[1, 0].set_xlabel('Identity')
    axs[1, 0].set_ylabel('Median')
    
    axs[1, 1].bar(identities_list, std_devs)
    axs[1, 1].set_title('Standard Deviation')
    axs[1, 1].set_xlabel('Identity')
    axs[1, 1].set_ylabel('Standard Deviation')
    
    axs[2, 0].bar(identities_list, mins)
    axs[2, 0].set_title('Min')
    axs[2, 0].set_xlabel('Identity')
    axs[2, 0].set_ylabel('Min')
    
    axs[2, 1].bar(identities_list, maxs)
    axs[2, 1].set_title('Max')
    axs[2, 1].set_xlabel('Identity')
    axs[2, 1].set_ylabel('Max')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output+'_epoch'+str(epoch)+'.png') #R100_epoch50.png
    return sorted_identities_list



def Stat(values, identities, output, epoch):
    stats = {}
    # Unique identities
    unique_identities = np.unique(identities)
    for identity in unique_identities:
        # Filter values corresponding to the current identity
        identity_values = values[identities == identity]
        # Calculate statistics
        min_val, max_val, avg_val = np.min(identity_values), np.max(identity_values), np.mean(identity_values)
        # Store statistics in a dictionary
        stats[identity] = {
        'frequency': len(identity_values),
        'mean': identity_values.mean(),
        'median': np.median(identity_values),
        'std': identity_values.std(),
        'min': identity_values.min(),
        'max': identity_values.max()
    }  
    # Display the results
    for identity in list(stats.keys()):
        print(f"Identity {identity}: {stats[identity]}")
    sorted_identities_list = StatPlot(stats, output, epoch)
    return sorted_identities_list, stats



import time
TL = []
delta = torch.ones(K) # Posteriror
save_delta = []
sorted_identities_list = []
filename = './VarNewResult/Finalcheck_nologR18'
for epoch in range(1, num_epochs+1):
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (x_batch, y_batch) in enumerate(train_loader, 1):
        start_time = time.time()
        loss = 1
        delta, pos, val = update_selection_variables(delta, x_batch, y_batch, model, criterion, optim, n, B, sigma_sq, r)
        if(epoch>950): # last 35 batch = 1050 mini batch 465 or 965
            save_delta.append([val.item(), pos])
        train_loss += loss
    if(epoch == 1000): #500 or 1000
        start_time = time.time()
        print(type(save_delta))
        DELTA = np.array(save_delta)
        XX, XY = Stat(np.array(DELTA[:, 0]), np.array(DELTA[:, 1]), filename, epoch)
        sorted_identities_list.append(XX)
    TL.append(train_loss)
    scheduler.step(train_loss)
    if train_loss < best_train_loss: best_train_loss, current_patience = train_loss, 0
    else:
        current_patience += 1
        print('current_patience =',current_patience)
        if current_patience >= patience:
            print(f'Early stopping after {epoch+1} epochs.')
            break    



import pickle
with open(filename+'.pkl', 'wb') as f:
    pickle.dump(sorted_identities_list, f)

with open(filename+'_delta.pkl', 'wb') as f:
    pickle.dump(np.array(save_delta), f)

with open(filename+'_stats.pkl', 'wb') as f:
    pickle.dump(XY, f)