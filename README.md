# Spatio-Temporal Informer(ST-Informer) Model
This repository contains the implementation of a **Scalable Informer-based deep learning architecture** for **water quality forecasting**. The model was primarily trained and tested on **multivariate water quality data** collected from various locations of the **16 major rivers in India**. Although demonstrated on the Indian Rivers dataset, the architecture is **generalizable**, **customizable** and can be applied to **other river systems or water quality monitoring datasets**.<br>
An interactive **Flask-based web interface** is also developed to deploy the trained model for real-time prediction and visualization.

## 📊Training Dataset Overview:
 - Source: Indian River water quality dataset.
 - Time Period: 2022-2024
 - Locations Count: 44
 - Water Quality parameters count: 23[BOD, COD, DO, pH, TDS,TC,FC,Nitrate and others]
 - Contextual parameters count: 3[Location. Year, Month]
 - Primary Target Parameter: Biochemical Oxygen Demand (BOD). 
 - The last column is the prediction target (BOD), while all other columns are input features.
 - Other Target Parameters:Chemical Oxygen Demand(COD),Dissolved Oxygen(DO) and Total Dissolved Solid(TDS)

## ⚙️ Dependencies

This project was developed and tested with the following dependencies:
```
- Python 3.10  
- torch >= 1.10.0  
- numpy >= 1.21.0  
- pandas >= 1.3.0  
- scikit-learn >= 1.0.0  
- matplotlib >= 3.4.0  
- joblib >= 1.0.0  
- openpyxl >= 3.0.0  
- python-dateutil >= 2.8.0
- scipy >= 1.7.0  
- flask >= 2.0.0  
```

## ⚙️ Installation
Clone the repository and install the required dependencies:
```
git clone https://github.com/DipeanDas/Spatio_Temporal_Informer_for_Indian_River_Water_Quality_Prediction.git
cd Spatio_Temporal_Informer_for_Indian_River_Water_Quality_Prediction
pip install -r requirements.txt 
```
## 👨‍💻 Contributors

**Dipean Dasgupta** (Class of 2025, Department of CSE, IIIT Vadodara, India)<br>
**Bishnu Prasad Sahoo** (Scientist, Forest Ecology and Climate Change Division, Forest Research Institute, Dehradun, India)<br>
**Pramit Mazumdar** (Assistant Professor, Department of CSE, IIIT Vadodara, India)<br>

