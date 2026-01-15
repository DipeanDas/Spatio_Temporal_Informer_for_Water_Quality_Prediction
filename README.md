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
