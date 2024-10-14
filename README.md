# Data-driven Rainfall Prediction at a Regional Scale: A Case Study with Ghana

## Description

Welcome to our project on **rainfall prediction in West Africa using deep learning techniques**. The challenge we are addressing stems from the difficulty that traditional numerical weather prediction (NWP) models face in providing accurate forecasts in this region, which is expected to experience more intense and unpredictable rainfall events due to climate change.

Reliable rainfall forecasts are crucial for effective water resource management, agriculture, and disaster preparedness. Our team at the **Foundations for Statistical Methods in the Environmental Sciences (FORMES)** group at **Boston University** is committed to developing innovative, data-driven approaches for this purpose. We leverage state-of-the-art machine learning techniques, specifically **U-Net convolutional neural network (CNN)** models, to predict 24-hour rainfall with 12-hour and 30-hour lead times.

Our goal is to significantly improve forecasting accuracy by utilizing extensive meteorological datasets, including the **ERA5 reanalysis dataset** and the **GPM-IMERG dataset**.

### Key Objectives:
1. Improve rainfall forecast accuracy for tropical regions, particularly Ghana.
2. Develop and apply machine learning models to achieve better lead-time predictions.
3. Enhance model interpretability by identifying important meteorological variables.
4. Combine data-driven and classical NWP techniques to optimize forecast performance.

### Main Achievements:
- Our 12-hour lead-time model matches or even outperforms traditional **ECMWF** forecasts.
- A novel statistical methodology has been developed to understand the role of different meteorological variables in driving precipitation in Ghana.
- We have shown that combining machine learning models with classical NWP methods enhances forecast accuracy.

Through this project, we aim to advance meteorological forecasting and contribute to improved water resource management, agriculture, and disaster preparedness in tropical regions.

## Repository Structure

- **`dataload.py`**: Prepares data processing and loads data. Called by `Train.py` and `Inference.ipynb`.
- **`dependency.py`**: Defines a set of functions required for executing the `Inference.ipynb` notebook.
- **`files.txt`**: Lists the ERA5 weather variables used by the project.
- **`Inference.ipynb`**: Notebook to load the models and validate the results.
- **`Latlon/`**: Folder containing latitude and longitude information for each grid.
- **`Models/`**: Directory for saved models (12-hour- and 30-hour lag).
- **`models64.py`**: Defines the U-Net model architecture used in the project.
- **`TestDate.txt`**: Specifies the date used to validate the results.
- **`Train.py`**: Script used to train the machine learning models.
- **`utils.py`**: Contains miscellaneous functions such as normalization.
- **`VariableSelection.py`**: Contains the code to evaluate the importance of its input predictors.
- **`download.py`**: To download the ERA5 data in netCDF format using the CDS API, which requires some prior setup on any user's machine (available in DownloadAndProcessData folder)
- **`run.py`**: To process the downloaded the ERA5 data. The script uses functions from **functions.py** to process the files into daily .npy arrays and save them to a new given location (available in DownloadAndProcessData folder)


## Requirements

Ensure the following dependencies are installed to run the code:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- xarray
- netCDF4
- scikit-learn
- OpenCV

Please install the required packages.

## How to Run

### 1. Preparing the Dataset

Before running the model, ensure the datasets are properly prepared. We are using two main datasets:
- **ERA5 Reanalysis Data**: This dataset contains global weather data from the European Centre for Medium-Range Weather Forecasts (ECMWF).
- **GPM-IMERG Dataset**: This is a satellite-based precipitation dataset from NASA.
- **TIGGE Dataset**: This database has established itself as a key window into the capability of state-of-the-art operational NWP models and has proven very useful to the research community.

#### Download the datasets:
- **ERA5 Reanalysis Data**: [Download Link](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **GPM-IMERG Dataset**: [Download Link](https://gpm.nasa.gov/data-access/downloads/gpm)
- **TIGGE Dataset**: [Download Link](https://apps.ecmwf.int/datasets/data/tigge/levtype=sfc/type=cf/)

Once downloaded, ensure the datasets are saved in the appropriate directory specified in the code and are in the correct format (NetCDF).

### 2. Training the Model

After the dataset is prepared, you can train the model using the following command:

```bash
python Train.py
```
### 3. Inference

After the model is successfully trained, you can move on to running inference, which involves using the trained model to make predictions on new or unseen data. 

```bash
jupyter notebook Inference.ipynb



