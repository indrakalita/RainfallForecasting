import numpy as np
import os
import re
import netCDF4 as nc4
from scipy.ndimage import zoom
from datetime import datetime

def day_of_year(date):
  """Calculates the day of the year for a given date string 20000601."""
  year = int(date[:4])
  month = int(date[4:6])
  day = int(date[6:])
  return datetime(year, month, day).timetuple().tm_yday, month

def LoadData(GPM_path, ERA_path, variable_path, ERA_zoom = True, GPM_zoom=False, GPM_desired_shape = (32, 32), ERA_desired_shape = (32, 32)):
    variables = [] # store all the variables
    with open(variable_path, 'r') as file:
        for line in file:
            variables.append(line.strip()) #Remove leading/trailing whitespaces and add folder name to the list
    #print(variables)
    files = sorted(os.listdir(GPM_path)) # Only one variable for GPM.
    count = 0
    for f in files:
        ##identify the date### and detect number of selected images
        file = f.split('.')[0] #  remove extension .nc4 # file name of GPM "IMERG_date.nc4"
        date = file.split('_')[1] # consider the date only
        N, month = day_of_year(date)
        if month in [1,2,3,4,5,6,7,8,9,10,11,12]:  #April to Sept #for dry---> [1,2,3,10,11,12] #for all---> [1,2,3,4,5,6,7,8,9,10,11,12]
            count = count + 1
    GPM_data = list()
    dateseq = list()
    if(ERA_zoom==True):
        ERA_data = np.zeros((count,len(variables)+3,ERA_desired_shape[0], ERA_desired_shape[1])) #+3 for cos, sin, and month
    else:
        ERA_data = np.zeros((count,len(variables)+3,30, 23))  #Dimension of ERA5 data (30,23)
    i = 0
    ################################# Main operaton for GPM ############################################
    for f in files:
        ##identify the date###
        file = f.split('.')[0] #  remove extension .nc4 # file name of GPM "IMERG_date.nc4"
        date = file.split('_')[1] # consider the date only
        N, month = day_of_year(date)
        if month in [1,2,3,4,5,6,7,8,9,10,11,12]: #April to Sept #for dry---> [1,2,3,10,11,12] #for all---> [1,2,3,4,5,6,7,8,9,10,11,12]
            dateseq.append(date)
            dta = nc4.Dataset(os.path.join(GPM_path,f))
            I = dta.variables['precipitationCal'][0,:].data #more smooth
            #I = dta.variables['HQprecipitation'][0,:].data # complex
            I = np.rot90(I) # To match with the ERA5 dataset 54X73-->73X54 #transpose will not work
            if(i==0):  # to read the lat long information of GPM and ERA5 data
                GPM_long = dta.variables['lon'][:].data
                GPM_lat = dta.variables['lat'][:].data[::-1] # read the latitute from last to first (11 is he first latitude)
                ERA_long = np.load('/projectnb/labci/Indrajit/Rainfall/data/ERA5/longitude_6h_2000-01-01_2023-09-07_filled.npy')[0]
                ERA_lat = np.load('/projectnb/labci/Indrajit/Rainfall/data/ERA5/latitude_6h_2000-01-01_2023-09-07_filled.npy')[0]
                GPM_cord = np.tile(GPM_lat, (len(GPM_long), 1)).T #To create 73X54 dimensional matrix with latitude only (GPM)
                ERA_cord = np.tile(ERA_lat, (len(ERA_long), 1)).T #To create 30X23 dimensional matrix with latitude only (ERA5)
            if(GPM_zoom==True):
                zoom_factors = (GPM_desired_shape[0] / I.shape[0], GPM_desired_shape[1] / I.shape[1]) # comment this if no zoom
                I = zoom(I, zoom_factors, order=3) #order=3 corresponds to cubic interpolation
            GPM_data.append(I)
            dta.close()
            ####################### Main operation for ERA5 #######################
            Cos = np.cos(2 * np.pi * N / 365) * GPM_cord
            Sin = np.sin(2 * np.pi * N / 365) * GPM_cord
            if(ERA_zoom==True):
                zoom_factors = (ERA_desired_shape[0] / Cos.shape[0], ERA_desired_shape[1] / Cos.shape[1]) # comment this if no zoom
                ERA_data[i,len(variables),:,:] = zoom(Cos, zoom_factors, order=3)
                ERA_data[i,len(variables)+1,:,:] = zoom(Sin, zoom_factors, order=3)
                ERA_data[i,len(variables)+2,:,:] = np.full((ERA_desired_shape[0], ERA_desired_shape[1]), month)  #the month of day
            else:
                zoom_factors = (30 / Cos.shape[0], 23 / Cos.shape[1]) # comment this if no zoom
                ERA_data[i,len(variables)+1,:,:] = zoom(Cos, zoom_factors, order=3)
                ERA_data[i,len(variables)+1,:,:] = zoom(Sin, zoom_factors, order=3)
                ERA_data[i,len(variables)+2,:,:] = np.full((30, 23), month)  #the month of day
            
            j = 0
            for name in variables:
                path = os.path.join(ERA_path,name)
                if re.search(r'\d{3}', name): # check for multi level variables that has the file name as "var_date.npy" 
                    fname = name+'_'+date+'.npy'
                else:                           # Other variables that has the file as "var_filled_data.npy" 
                    fname = name+'_filled_'+date+'.npy'
                #print(path)
                I = np.load(os.path.join(path,fname))
                I = I[1,:,:]  # Noon observation [0,1,2,3][12 Midnight, 6AM, 12Noon, 6PM]
                #print('ERA5',fname, I.shape,I)
                if(ERA_zoom==True):
                    zoom_factors = (ERA_desired_shape[0] / I.shape[0], ERA_desired_shape[1] / I.shape[1]) # comment this if no zoom
                    I = zoom(I, zoom_factors, order=3)  # order=3 corresponds to cubic interpolation # comment this if no zoom
                ERA_data[i,j,:,:] = I
                j = j + 1 # Go to next variable
            i = i + 1 # processiong of the first GPM image is done
    return np.array(GPM_data), np.transpose(ERA_data,(1,0,2,3)), [GPM_long, GPM_lat, ERA_long, ERA_lat], np.array(dateseq) #NXCXDXWXH
