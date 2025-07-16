import isodisreg 
from isodisreg import idr
import properscoring as ps
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
    path = '/projectnb/labci/NAS/Project/ShortTermRainfallForecast_Ghana/version1/model&data/'
    print("Which model you want to load")
    print("1: Model Mid-Night (12AM)")
    print("2: Model Morning (6AM)")
    print("3: Model Noon (12PM)")
    print("4: Model Night (6PM)")
    M = input("Please provide your input")
    if M == '1':
        norm = 'norm'
        model_path = os.path.join(path,'data0/model/Model0_L1Loss_Timezone0.pth') #data0/model/Model0_L1Loss_Timezone0.pth
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device) #57
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'data0/Model0_Testloader_57V_Eznorm_Gznorm_MSEL.pkl') #data0/Model0_Testloader_57V_Eznorm_Gznorm_MSEL.pkl
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        test_loader = loaded_data['test_loader']
        Gmean = loaded_data['Gmean']
        Gstd = loaded_data['Gstd']
        print(Gmean,Gstd)
        return model, test_loader, Gmean, Gstd, norm
    elif M == '2':
        norm = 'norm'
        model_path = os.path.join(path,'data1/model/Model1_L1Loss_Timezone1.pth')   #Model0_MSE_Timezone1.pth
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'data1/Model1_Testloader_57V_Eznorm_Gnorm_L1.pkl')
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        test_loader = loaded_data['test_loader']
        Gmean = loaded_data['Gmean']
        Gstd = loaded_data['Gstd']
        print(Gmean,Gstd)
        return model, test_loader, Gmean, Gstd, norm
    elif M == '3':
        norm = 'norm'
        model_path = os.path.join(path,'data2/model/Model1_L1Loss_Timezone2.pth') #Model0_SmoothL1Loss_Timezone2.pth
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'data2/Model1_Testloader_57V_Eznorm_Gnorm_L1.pkl')
        with open(data_path, 'rb') as f:
            loaded_data = pickle.load(f)
        test_loader = loaded_data['test_loader']
        Gmean = loaded_data['Gmean']
        Gstd = loaded_data['Gstd']
        print(Gmean,Gstd)
        return model, test_loader, Gmean, Gstd, norm
    elif M == '4':
        norm = 'norm'
        model_path = os.path.join(path,'data3/model/Model1_L1Loss_Timezone3.pth') #Model1_L1Loss_Timezone3.pth, Model0_MSE_Timezone3.pth
        print("Loading model...")
        model = models.UNet(in_channels=57, out_channels=64, n_class=1, kernel_size=3, padding=1, stride=1).to(device)
        model.load_state_dict(torch.load(model_path))
        print("Loading the required data")
        data_path = os.path.join(path,'data3/Model1_Testloader_57V_Eznorm_Gnorm_L1.pkl') #
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
# Evaluation using TIGG data. Do not run this cell for models. It is only for TIGG data
def Evaluation(GPM_path, TIG_path, GPM_zoom=True, GPM_desired_shape = (64, 64), TIG_zoom=True, TIG_desired_shape = (64, 64)):
    files = sorted(os.listdir(TIG_path+'tiggePerturbedEnsemble')) # or tiggePerturbedEnsemble tiggeControl
    TIG_pturb, TIG_control, GPM_data  = list(), list(), list()
    D, P = list(), list()
    for f in files:
        # ---------------------------------reading pturb---------------------------------
        dta = nc4.Dataset(os.path.join(TIG_path, 'tiggePerturbedEnsemble/', f))  # reading pturb
        I1 = np.mean(dta.variables['tp'], axis=1) # dimension (3, 50, 30, 23)--> mean (3,30,23)
        I1 = I1[2,:,:] - I1[1,:,:] # difference between the third and second values gives the 24h rainfall forecast with 18h lead-time
        if(TIG_zoom==True):
            zoom_factors = (TIG_desired_shape[0] / I1.shape[0], TIG_desired_shape[1] / I1.shape[1])
            I1 = zoom(I1, zoom_factors, order=3)
        TIG_pturb.append(I1)
        dta.close()
        # ---------------------------------reading control---------------------------------
        dta = nc4.Dataset(os.path.join(TIG_path, 'tiggeControl/', f))  # reading pturb
        I1 = dta.variables['tp'] # dimension (3, 30, 23)
        I1 = I1[2,:,:] - I1[1,:,:] # difference between the third and second values gives the 24h rainfall forecast with 18h lead-time
        if(TIG_zoom==True):
            zoom_factors = (TIG_desired_shape[0] / I1.shape[0], TIG_desired_shape[1] / I1.shape[1])
            I1 = zoom(I1, zoom_factors, order=3)
        TIG_control.append(I1)
        dta.close()
        # ---------------------------------reading GPM---------------------------------
        file = f.split('.')[0]
        date = file.split('_')[3]
        f_name = 'IMERG_'+date+'.nc4'
        dta = nc4.Dataset(os.path.join(GPM_path,f_name))
        I2 = dta.variables['precipitationCal'][0,:].data #more smooth
        I2 = np.rot90(I2) #
        if(GPM_zoom==True):
            zoom_factors = (GPM_desired_shape[0] / I2.shape[0], GPM_desired_shape[1] / I2.shape[1]) # comment this if no zoom
            I2 = zoom(I2, zoom_factors, order=3)
        GPM_data.append(I2)
        dta.close()
        ##################Operation for test images using model#######################
        targetdate, position = find_date_position('TestDate.txt', int(date))
        P.append(position), D.append(targetdate)
    return np.array(GPM_data), np.array(TIG_pturb), np.array(TIG_control), np.array(P), np.array(D)

##### Random Evaluation####
def generate_relevant_dates(target_date):
    start_date = datetime(2000, 6, 1)
    end_date = datetime(2021, 9, 30)
    relevant_dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Calculate offset days around the target_date
        for offset in [-2, -1, 0, 1, 2]:
            relevant_date = target_date.replace(year=current_date.year) + timedelta(days=offset) #yesterday = today + timedelta(days=-1)
            if start_date <= relevant_date <= end_date:
                relevant_dates.append(relevant_date.date())
        current_date = current_date.replace(year=current_date.year + 1)
    return relevant_dates

def generate_relevant_dates_bymonth(target_date):
    start_date = datetime(2001, 1, 1)
    end_date = datetime(2006, 12, 31)
    relevant_dates = []
    month = target_date.month  # Get the month from the target_date
    day = target_date.day    # Get the day from the target_date
    
    for year in range(2001, target_date.year): # Loop through the years before the target year
        # Create a datetime for each date of the same month and year
        for day_in_month in range(1, 32): # maximum 31 days
            try:
                # Construct a date with the same month and day (may fail for invalid days in some months)
                relevant_date = datetime(year, month, day_in_month)
                if relevant_date.month == month:  # Only keep valid dates for the month
                    relevant_dates.append(relevant_date)
            except ValueError:
                continue  # Skip invalid days like 31st in February
    return relevant_dates


def RandomEvaluation(GPM, TargetDate, GPM_zoom=False, GPM_desired_shape = (32, 32)):
    Avg_data, Indv_data, GTR = [], [], []
    TargetDate_str = TargetDate.astype(str)
    target_dates = np.array([datetime.strptime(date, '%Y%m%d') for date in TargetDate_str])
    for target_date in target_dates:
        gt_date = target_date.strftime('%Y%m%d')
        #print('Ground truth date=',gt_date)
        relevant_dates = generate_relevant_dates(target_date)
        GPM_data = []
        for r in relevant_dates:
            R = r.strftime('%Y%m%d')
            if(R<gt_date):
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
        #print(np.array(GPM_data).shape)
        Indv_data.append(GPM_data) #[0;104]
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
    return np.array(Avg_data), Indv_data, np.array(GTR)

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

##### Prepare the netCDF file
# Save the data into netcdf format
# Create a new NetCDF file
def prepareCDF(ground_truth, model_prediction, dates, latitude, longitude, file_path):
    dataset = nc4.Dataset(file_path, 'w', format='NETCDF4_CLASSIC')
    
    # Create the dimensions of the data
    dataset.createDimension('sample', ground_truth.shape[0])
    dataset.createDimension('x', ground_truth.shape[1])
    dataset.createDimension('y', ground_truth.shape[2])
    dataset.createDimension('date_str_len', 8)  # Assuming date strings are of length 8 (YYYYMMDD)
    dataset.createDimension('lat', len(latitude))
    dataset.createDimension('lon', len(longitude))
    
    # Create variables
    ground_truth_var = dataset.createVariable('ground_truth', np.float32, ('sample', 'x', 'y'))
    model_prediction_var = dataset.createVariable('model_prediction', np.float32, ('sample', 'x', 'y'))
    dates_var = dataset.createVariable('dates', 'S1', ('sample', 'date_str_len'))
    latitude_var = dataset.createVariable('latitude', np.float32, ('lat',))
    longitude_var = dataset.createVariable('longitude', np.float32, ('lon',))
    
    # Add data to variables
    ground_truth_var[:, :, :] = ground_truth
    model_prediction_var[:, :, :] = model_prediction
    
    # Handling string data for dates
    dates_char = nc4.stringtochar(dates.astype('S8'))  # Convert to fixed size strings of length 8
    dates_var[:, :] = dates_char
    
    # Add latitude and longitude data
    latitude_var[:] = latitude
    longitude_var[:] = longitude
    
    # Add attributes (optional)
    ground_truth_var.units = 'mm'  # Replace with actual unit
    model_prediction_var.units = 'mm'  # Replace with actual unit
    dates_var.description = 'Dates for each sample'
    latitude_var.units = 'degrees_north'
    longitude_var.units = 'degrees_east'
    
    # Close the dataset
    dataset.close()

# Verify the netCDF file
# Verify the data
path = '/projectnb/labci/Indrajit/Rainfall_New/'
file_path = os.path.join(path, 'Testing.nc') #climatology_Model.nc, TIGGPrediction.nc, Night_model.nc, Noon_model.nc, Morning_model.nc, MidNight_model.nc
def load_data(file_path):
    dataset = nc4.Dataset(file_path, 'r')
    ground_truth = dataset.variables['ground_truth'][:]
    model_prediction = dataset.variables['model_prediction'][:]
    dates = nc4.chartostring(dataset.variables['dates'][:])
    latitude = dataset.variables['latitude'][:]
    longitude = dataset.variables['longitude'][:]
    dataset.close()
    return ground_truth, model_prediction, dates, latitude, longitude

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

# Other calculations
'''
Threat Score (TS)/Critical success index (CSI): This metric focuses on correctly predicted "events" (areas with precipitation above a threshold).
It considers both hits (correctly predicted precipitation) and false alarms (predicted precipitation that didn't occur).
Higher the better
False Alarm Ratio (FAR): This metric focuses on the number of false alarms relative to the total number of predicted events.
Lower FAR indicates fewer incorrect precipitation predictions.
Probability of detection (POD): Fraction of event correctly forecasted
Equitable Threat score (ETS): Generic form of CSI with random forecast as reference
'''
def score(ground_truth, forecast, threshold):
    TP, FP, FN, TN = 0, 0, 0, 0
    for sample in range(len(ground_truth)):
        gt_mask = ground_truth[sample] > threshold
        forecast_mask = forecast[sample] > threshold
        TP += np.sum(np.logical_and(forecast_mask, gt_mask))
        FP += np.sum(np.logical_and(forecast_mask, np.logical_not(gt_mask)))
        FN += np.sum(np.logical_and(np.logical_not(forecast_mask), gt_mask))
        TN += np.sum(np.logical_and(np.logical_not(forecast_mask), np.logical_not(gt_mask)))
    return TP, FP, FN, TN
def score_calculate(Target, Data, thresholds, Print=False):
    avg_ts, avg_fpr, avg_pod, avg_f1, avg_acc = 0, 0, 0, 0, 0
    TS, FPR, POD, F1 = np.empty(len(thresholds)),np.empty(len(thresholds)),np.empty(len(thresholds)), np.empty(len(thresholds))
    ACC = np.empty(len(thresholds))
    i = 0
    for threshold in thresholds:  # You can try different thresholds 
        TP, FP, FN, TN = score(Target, Data, threshold)
        ts, fpr, pod = TP/(TP + FP + FN), TP/(TP + FP), TP/(TP+FN)  #CSI, Precision, Recall
        f1 = 2*(fpr*pod)/(fpr+pod)
        acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 0  # Accuracy
        #print('TS or CSI, Precision, Recall, F1 =', ts, fpr, pod, f1)
        TS[i], FPR[i], POD[i], F1[i], ACC[i] = ts, fpr, pod, f1, acc
        
        i =  i + 1
        avg_ts += ts
        avg_fpr +=fpr
        avg_pod +=pod
        avg_f1 +=f1
        avg_acc += acc
    if(Print==True):
        print("Average TS/CSI, Precision, Recall (for all the different thresholds):", avg_ts/len(thresholds),avg_fpr/len(thresholds),avg_pod/len(thresholds), avg_f1/len(thresholds))
    return avg_ts/len(thresholds),avg_fpr/len(thresholds),avg_pod/len(thresholds), avg_f1/len(thresholds), avg_acc/len(thresholds)


##---------------------New function--------------------#
def Indv_TIGG(TIG_path, TIG_zoom=True, TIG_desired_shape = (64, 64)):
    files = sorted(os.listdir(TIG_path+'tiggePerturbedEnsemble'))
    TIG_pturb = []
    for f in files:
        dta = nc4.Dataset(os.path.join(TIG_path, 'tiggePerturbedEnsemble/', f))  # reading pturb
        I1 = dta.variables['tp']
        I1 = I1[2,:,:,:] - I1[1,:,:,:]
        zoom_factors = (1, TIG_desired_shape[0] / I1.shape[1], TIG_desired_shape[1] / I1.shape[2])
        I1 = zoom(I1, zoom_factors, order=3)
        TIG_pturb.append(I1)
        dta.close()
    return TIG_pturb

def CRPS_Ens(pr, gt):
    crps_map = np.zeros((64, 64))
    for i in range(64):
        for j in range(64): # For each pixel (i, j), compute CRPS across all samples
            crps_map[i, j] = np.mean(ps.crps_ensemble(gt[:, i, j], pr[:, :, i, j]))
    return crps_map