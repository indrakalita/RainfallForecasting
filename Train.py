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




#Normalization
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
model = ModelSelection(1,fine_tune=False) # 1 basic UNET, 2 Standard UNET, 3 Pre-trained UNET, fine_tune is for 3rd model only
# Verify the output of the model is same with the target
x = torch.randn(1,X.shape[1], 64, 64).to(device)
pred = model(x)
print('Model compiled successfully and the size of the output is: ',pred.shape)


#Hyparameters
from torch.optim import lr_scheduler
from torch.optim import Adam
learning_rate, num_epochs, step_size, gamma = 1e-4, 10000, 70, 0.1
optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #0.00001
criterion = nn.L1Loss(reduction='sum')#nn.MSELoss()#nn.BCELoss(reduction='sum')
scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=gamma, patience=step_size, verbose=True)
best_train_loss, current_patience, patience = np.inf, 0, 120


import random
def occlusion(batch, region_size=(5, 5), num_regions=2):
    batch_size, channels, height, width = batch.size()
    # Iterate over each image in the batch
    for i in range(batch_size):
        # Select random positions for the regions to be removed
        for _ in range(num_regions):
            top = random.randint(0, height - region_size[0])
            left = random.randint(0, width - region_size[1])
            # Remove the region from the image
            batch[i, :, top:top+region_size[0], left:left+region_size[1]] = 0
    return batch
augsize = 1 # no augmentation, >1 augmentation



import torch.nn.functional as F
TL = []
for epoch in range(1, num_epochs+1):
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):
        for i in range(0,augsize):
            if(i>0): input = occlusion(input, region_size=(2, 2), num_regions=100)
            input, target = input.to(device),target.to(device)
            output = model(input)
            loss = criterion(output.flatten(), target.flatten()) # Standard MSE loss, LL1 loss
            average_loss = loss/len(input)
            average_loss.backward()                                            
            optim.step()                                               
            optim.zero_grad()                                           
            train_loss = train_loss + loss.item()/len(input)
    train_loss = train_loss/(augsize*len(train_loader.dataset))
    if(epoch%1000==0): #Check intermediate output after 1000 epochs
        T,P = target[0,0].cpu().detach().numpy(), output[0,0].cpu().detach().numpy()
        ut.visualize(GT=T, PR=P)
    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:
            input, target = input.to(device),target.to(device)
            output = model(input)                  
            loss = criterion(output.flatten(), target.flatten()) # Standard MSE loss, LL1 loss
            val_loss = val_loss + loss.item()/len(input)                        
    val_loss /= len(val_loader.dataset)
    TL.append(train_loss)
    print("Epoch:{} Training Loss:{:.20f} Validation Loss:{:.20f}\n".format(epoch, train_loss, val_loss))
    if(epoch%1000==0): #Check intermediate output after 1000 epochs
        T,P = target[0,0].cpu().detach().numpy(), output[0,0].cpu().detach().numpy()
        ut.visualize(GT1=T, PR1=P)
    #Early stopping criteria
    scheduler.step(train_loss)
    if(epoch>1000 and epoch%1000==0):
        print('Saving the best model at epoch:', epoch)
        torch.save(model.state_dict(), 'model_path')
        #Example path (already exist please change the path):
        #'/Path to save/Model2_MSE_Timezone1_epoch_'+str(epoch)+'.pth'
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        current_patience = 0
    else:
        current_patience += 1
        print('current_patience =',current_patience)
        if current_patience >= patience:
            print(f'Early stopping after {epoch+1} epochs.')
            break



#Run this cell only if you want to save the model
torch.save(model.state_dict(), 'Model_path')
#/Path to save/Model2_MSE_Timezone1.pth (example path)



TL = np.array(TL)
epochs = range(1, len(TL) + 1)
plt.plot(epochs, TL, label='Training Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('Image.png')
#/Path to save/Model2_MSE_Timezone1.png (image path as example)

