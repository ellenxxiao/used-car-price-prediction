# used-car-price-prediction

## Introduction 
When I was helping my best friend shopping an used car, we kept thinking and asking if the price is reasonable, if we can have a better price. It will be helpful if we have a model that can predict the price based on other used car, then we could determine if the car price reasonable in a minute. As a data science learner, I know there are machine learning algorithms that may work. 

In this project, I applied machine learning algorithms to predict used car price.  

[Here](https://github.com/ellenxxiao/used-car-price-prediction/blob/master/Used%20cars.ipynb) is my project.

## Table of Content
- [DataSet](#Dataset) 
- [Data Preprocessing](#Data_Preprocessing) 
- [Exploratory Data Analysis](#Exploratory_Data_Analysis)  
- [Models](#Models)
- [Results](#Results)

## Dataset
The data source is downloaded <a href="https://www.kaggle.com/orgesleka/used-cars-database" target="_blank">here</a> from Kaggle

## Data_Preprocessing
<a href="https://github.com/ellenxxiao/used-car-price-prediction/blob/master/Data%20Preprocessing.py" target="_blank">Data Preprocessing</a> contains 4 steps:
1. Remove outliers (1.5 IQR Rules) 
-Example
```shell
lower_bound = df['yearOfRegistration'].quantile(.25)-(df['yearOfRegistration'].quantile(.75)-df['yearOfRegistration'].quantile(.25))*1.5
upper_bound = df['yearOfRegistration'].quantile(.75)+(df['yearOfRegistration'].quantile(.75)-df['yearOfRegistration'].quantile(.25))*1.5

lower_bound = df['powerPS'].quantile(.25)-(df['powerPS'].quantile(.75)-df['powerPS'].quantile(.25))*1.5
upper_bound = df['powerPS'].quantile(.75)+(df['powerPS'].quantile(.75)-df['powerPS'].quantile(.25))*1.5

lower_bound = df['kilometer'].quantile(.25)-(df['kilometer'].quantile(.75)-df['kilometer'].quantile(.25))*1.5
upper_bound = df['kilometer'].quantile(.75)+(df['kilometer'].quantile(.75)-df['kilometer'].quantile(.25))*1.5

lower_bound = df['price'].quantile(.25)-(df['price'].quantile(.75)-df['price'].quantile(.25))*1.5
upper_bound = df['price'].quantile(.75)+(df['price'].quantile(.75)-df['price'].quantile(.25))*1.5
```
2. Drop unnecessary columns
3. Fill missing values  
-Example
```shell
df_update['fuelType'].fillna(value='not-declared', inplace=True)
df_update['gearbox'].fillna(value='not-declared', inplace=True)
```
4. Create new features  
-Example
```shell
df_update['daysOnline'] = pd.to_datetime(df_update['lastSeen'])-pd.to_datetime(df_update['dateCreated'])
df_update['daysOnline'] = df_update['daysOnline'].dt.days+1
```
## Exploratory_Data_Analysis
Code is <a href="https://github.com/ellenxxiao/used-car-price-prediction/blob/master/EDA.py" target="_blank">here</a>
1. The distribution of features  
-Example

![Capture](https://user-images.githubusercontent.com/26680796/88431685-d80b6480-cdc8-11ea-95e4-c98613e0674b.png)
2. The price changed in years
3. Price distribution among features

## Models
The model I applied are Linear Regression and RandomForest Regression. Code is <a href="https://github.com/ellenxxiao/used-car-price-prediction/blob/master/Models.py" target="_blank">here</a>
1. Encode data and standardize
2. PCA
3. Linear Regression
4. RandomForest Regression
5. Feature Importance

## Results
Testing Accuracy: 87.7%

