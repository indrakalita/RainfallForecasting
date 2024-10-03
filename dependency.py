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
import netCDF4 as nc4
from scipy.ndimage import zoom
from datetime import datetime
import utils as ut
import models64 as models
from datetime import datetime, timedelta
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

# To obtain the date and its position
def find_date_position(file_path, target_date):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        dates = [int(line.strip()) for line in lines]
    try:
        position = dates.index(target_date)
        return target_date, position
    except ValueError:
        return -1


#Load model
def load_model():
    path = '/Models/'
    print("Which model you want to load")
    print("1: Model Mid-Night (30 hours lag)")
    print("2: Model Night (12 hours lag)")
    M = input("Please provide your input")
    if M == '1':
        norm = 'norm'
        model_path = os.path.join(path,'Model0_L1Loss_Timezone0.pth')
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device) #57
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'/Put the testing file/')
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        test_loader = loaded_data['test_loader']
        Gmean = loaded_data['Gmean']
        Gstd = loaded_data['Gstd']
        print(Gmean,Gstd)
        return model, test_loader, Gmean, Gstd, norm
    elif M == '2':
        norm = 'norm'
        model_path = os.path.join(path,'Model1_L1Loss_Timezone3.pth')
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'/Put the testing file/') #
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        test_loader = loaded_data['test_loader']
        Gmean = loaded_data['Gmean']
        Gstd = loaded_data['Gstd']
        print(Gmean,Gstd)
        return model, test_loader, Gmean, Gstd, norm
    else:
        print("Invalid selection. Please enter 1 to 4.")

### Evaluation for TIGG data only######################
def Evaluation(GPM_path, TIG_path, GPM_zoom=True, GPM_desired_shape = (64, 64), TIG_zoom=True, TIG_desired_shape = (64, 64)):
    files = sorted(os.listdir(TIG_path))
    TIG_data = list()
    GPM_data = list()
    D = list()
    P = list()
    for f in files:
        dta = nc4.Dataset(os.path.join(TIG_path,f))
        I1 = dta.variables['tp']
        #print(I1[0,:,:])
        I1 = I1[7,:,:] - I1[3,:,:] # 12PM,6PM,12AM,#6AM,12PM,6PM,12AM#,6AM,12PM
        if(TIG_zoom==True):
            zoom_factors = (TIG_desired_shape[0] / I1.shape[0], TIG_desired_shape[1] / I1.shape[1])
            I1 = zoom(I1, zoom_factors, order=3)
        dta.close()
        # Get the date
        file = f.split('.')[0]
        date = file.split('_')[3]
        date = date.replace("-", "")
        #D.append(date)
        f_name = 'IMERG_'+date+'.nc4'
        dta = nc4.Dataset(os.path.join(GPM_path,f_name))
        I2 = dta.variables['precipitationCal'][0,:].data #more smooth
        #print(I2.shape)
        I2 = np.rot90(I2) #
        if(GPM_zoom==True):
            zoom_factors = (GPM_desired_shape[0] / I2.shape[0], GPM_desired_shape[1] / I2.shape[1]) # comment this if no zoom
            I2 = zoom(I2, zoom_factors, order=3)
        dta.close()
        ##################Operation for test images using model#######################
        targetdate, position = find_date_position('TestDate.txt', int(date))
        #print(position, date)
        P.append(position), D.append(targetdate)
        TIG_data.append(I1), GPM_data.append(I2)
    return np.array(GPM_data), np.array(TIG_data), np.array(P), np.array(D)

##### Random Evaluation####
def generate_relevant_dates(target_date):
    start_date = datetime(2000, 6, 1)
    end_date = datetime(2021, 9, 30)
    relevant_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Calculate offset days around the target_date
        for offset in [-2, -1, 0, 1, 2]:
            relevant_date = target_date.replace(year=current_date.year) + timedelta(days=offset)
            if start_date <= relevant_date <= end_date:
                relevant_dates.append(relevant_date.date())
        current_date = current_date.replace(year=current_date.year + 1)
    return relevant_dates


def RandomEvaluation(GPM, TargetDate, GPM_zoom=False, GPM_desired_shape = (32, 32)):
    Avg_data, GTR = [], []
    TargetDate_str = TargetDate.astype(str)
    target_dates = np.array([datetime.strptime(date, '%Y%m%d') for date in TargetDate_str])
    for target_date in target_dates:
        gt_date = target_date.strftime('%Y%m%d')
        #print('Ground truth date=',gt_date)
        relevant_dates = generate_relevant_dates(target_date)
        GPM_data = []
        for r in relevant_dates:
            R = r.strftime('%Y%m%d')
            if(R!=gt_date):
                filename = 'IMERG_'+R+'.nc4'
                #print('Random dates=',filename)
                dta = nc4.Dataset(os.path.join(GPM,filename))
                I = dta.variables['precipitationCal'][0,:].data #more smooth
                I = np.rot90(I) #
                if(GPM_zoom==True):
                    zoom_factors = (GPM_desired_shape[0] / I.shape[0], GPM_desired_shape[1] / I.shape[1]) # comment this if no zoom
                    I = zoom(I, zoom_factors, order=3)
                GPM_data.append(I)
                dta.close()
        #Take the mean of all the images
        avg_data = np.mean(np.array(GPM_data), axis=0)
        Avg_data.append(avg_data)
        #For the ground truth
        filename = 'IMERG_'+gt_date+'.nc4'
        dta = nc4.Dataset(os.path.join(GPM,filename))
        Random_GT = dta.variables['precipitationCal'][0,:].data #more smooth
        Random_GT = np.rot90(Random_GT)
        if(GPM_zoom==True):
            zoom_factors = (GPM_desired_shape[0] / Random_GT.shape[0], GPM_desired_shape[1] / Random_GT.shape[1]) # comment this if no zoom
            Random_GT = zoom(Random_GT, zoom_factors, order=3)
        dta.close()
        GTR.append(Random_GT)
    return np.array(Avg_data), np.array(GTR)

#### Select the dates based on the dates available on TIGG data
def extract_ordered_subset_from_loader(data_loader, indices):
    subset_inputs = []
    subset_targets = []
    indices_set = set(indices)
    data_dict = {idx: None for idx in indices}
    
    for idx, (input, target) in enumerate(data_loader):
        if idx in indices_set:
            data_dict[idx] = (input, target)
        if all(value is not None for value in data_dict.values()):
            break
    
    for idx in indices:
        input, target = data_dict[idx]
        subset_inputs.append(input)
        subset_targets.append(target)
    
    subset_inputs = torch.cat(subset_inputs, dim=0)
    subset_targets = torch.cat(subset_targets, dim=0)
    
    return subset_inputs, subset_targets


#####Evaluation functions
#MAE calculation
def mae(Target, Data):
    L = 0
    for i in range(0,len(Target)):
        loss = np.mean(np.abs(Target[i] - Data[i]))
        L = L + loss
        #visualize(i, L, GPM=Target[i], Pred=Data[i])
    L = L/(i+1)
    return L
# Variance calculator
def variance(Target, Data):
    variances = np.zeros(len(Data))  # Array to store variances for all images
    D_array = []
    for i in range(len(Data)):
        difference_array = Data[i] - Target[i]
        D_array.append(difference_array) #pixel wise variance are stored in a list
        variance = np.var(difference_array) # variance of each sample/image
        variances[i] = variance
    average_variance1 = np.mean(variances) # average of variance
    print("Average variance of the differences across all images:", average_variance1)
    D_array = np.array(D_array)
    average_variance2 = np.var(D_array)
    print("Variance of the differences across all pixels:", average_variance2)
    plt.hist(D_array.flatten(), bins=200)
    plt.xlabel('Difference value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Variance of all the pixels available in the test set')
    plt.show()
#Compute histogram and calculate frequency count
def compute_histogram(Target, Data, bin_edges, Print=False, Hist=False):
    # Compute the difference array
    D_array = [Data[i] - Target[i] for i in range(len(Data))]
    D_array = np.array(D_array)
    
    # Flatten the difference array
    flattened_D_array = D_array.flatten()
    
    # Compute the histogram with custom bin edges
    hist, bin_edges = np.histogram(flattened_D_array, bins=bin_edges)
    if(Print==True):
        for i in range(len(hist)):
            print(f"Range: ({bin_edges[i]}, {bin_edges[i+1]}) - Frequency: {hist[i]}")
    if(Hist==True):
        plt.hist(flattened_D_array, bins=bin_edges)
        plt.xlabel('Difference value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Variance of all the pixels available in the test set')
        plt.show()
    
    return hist, bin_edges

# Evaluation
def score(ground_truth, forecast, threshold):
    TP,FP,FN = 0,0,0
    for sample in range(len(ground_truth)):
        gt_mask = ground_truth[sample] > threshold
        forecast_mask = forecast[sample] > threshold
        TP += np.sum(np.logical_and(forecast_mask, gt_mask))
        FP += np.sum(np.logical_and(forecast_mask, np.logical_not(gt_mask)))
        FN += np.sum(np.logical_and(np.logical_not(forecast_mask), gt_mask))
    return TP, FP, FN
def score_calculate(Target, Data, thresholds, Print=False):
    avg_ts, avg_fpr, avg_pod, avg_f1 = 0, 0, 0, 0
    TS, FPR, POD, F1 = np.empty(len(thresholds)),np.empty(len(thresholds)),np.empty(len(thresholds)), np.empty(len(thresholds))
    i = 0
    for threshold in thresholds:  # You can try different thresholds 
        TP, FP, FN = score(Target, Data, threshold)
        ts, fpr, pod = TP/(TP + FP + FN), TP/(TP + FP), TP/(TP+FN)  #CSI, Precision, Recall
        f1 = 2*(fpr*pod)/(fpr+pod)
        #print('TS or CSI, Precision, Recall, F1 =', ts, fpr, pod, f1)
        TS[i], FPR[i], POD[i], F1[i] = ts, fpr, pod, f1
        i =  i + 1
        avg_ts += ts
        avg_fpr +=fpr
        avg_pod +=pod
        avg_f1 +=f1
    if(Print==True):
        print("Average TS/CSI, Precision, Recall (for all the different thresholds):", avg_ts/len(thresholds),avg_fpr/len(thresholds),avg_pod/len(thresholds), avg_f1/len(thresholds))
    return avg_ts/len(thresholds),avg_fpr/len(thresholds),avg_pod/len(thresholds), avg_f1/len(thresholds)