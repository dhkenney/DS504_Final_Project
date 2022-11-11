# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:29:18 2022

@author: David
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
import scipy as scp
from scipy.optimize import NonlinearConstraint, Bounds



data = pd.read_excel(r"C:\Users\David\OneDrive - Worcester Polytechnic Institute (wpi.edu)\Documents\WPI\Classwork\Year 3\DS504 - Big Data\Final Project\Full Data.xlsx",sheet_name = 'Sheet1')

unique_states = data.State.unique()

states = np.zeros((len(data),len(unique_states)))

for ii in range(0,len(data)):
    print(ii)
    for jj in range(0,len(unique_states)):
        if data['State'][ii] == unique_states[jj]:
            states[ii,jj] = 1
            
states = pd.DataFrame(states,columns = ['Alaska','Alabama','Arkansas','Arizona','California','Colorado','Connecticut','D.C.','Delaware','Florida','Georgia','Hawaii','Iowa','Idaho','Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesotta','Missouri','Missisippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey','New Mexico','Nevada','New York','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming','Puerto Rico'])


fuel_types = data['Plant Primary Fuel'].unique()

fuels = np.zeros((len(data),len(fuel_types)))

for ii in range(0,len(data)):
    print(ii)
    for jj in range(0,len(fuel_types)):
        if data['Plant Primary Fuel'][ii] == fuel_types[jj]:
            fuels[ii,jj] = 1
            
fuels = pd.DataFrame(fuels,columns = fuel_types)           
            
data = pd.concat([data['Year'],states,data.iloc[:,2],data.iloc[:,4:len(data.columns)]],axis = 1)
 

unique_states = ['Alaska','Alabama','Arkansas','Arizona','California','Colorado','Connecticut','D.C.','Delaware','Florida','Georgia','Hawaii','Iowa','Idaho','Illinois','Indiana','Kansas','Kentucky','Louisiana','Massachusetts','Maryland','Maine','Michigan','Minnesotta','Missouri','Missisippi','Montana','North Carolina','North Dakota','Nebraska','New Hampshire','New Jersey','New Mexico','Nevada','New York','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Virginia','Vermont','Washington','Wisconsin','West Virginia','Wyoming','Puerto Rico']

years = data.Year.unique()
    
avg_nox = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_so2 = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_co2 = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_gen = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_pgc = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_heat_input = pd.DataFrame(np.zeros((len(unique_states),len(years))))
avg_net_generation = pd.DataFrame(np.zeros((len(unique_states),len(years))))


for ii in range(0,len(unique_states)):
    for jj in range(0,len(years)):
        inter = data[(data[unique_states[ii]] == 1) & (data['Year'] == years[jj])].mean()
        
        avg_nox.iloc[ii,jj] = inter.loc['Annual NOx Emissions (tons)']
        avg_so2.iloc[ii,jj] = inter.loc['SO2 Emissions (tons)']
        avg_co2.iloc[ii,jj] = inter.loc['CO2 Emissions (tons)']
        avg_gen.iloc[ii,jj] = inter.loc['Number of Generators']
        avg_pgc.iloc[ii,jj] = inter.loc['Plant Generator Capacity (MW)']
        avg_heat_input.iloc[ii,jj] = inter.loc['Annual Heat Input (MMBTU)']
        avg_net_generation.iloc[ii,jj] = inter.loc['Annual Net Generation (MWh)']
    
"""Model Development"""

X = pd.concat([data.iloc[:,53:58],data.iloc[:,61:len(data.columns)]],axis=1)
y = data['CO2 Emissions (tons)']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=42)


model1 = LinearRegression()
model2 = Ridge()
model3 = Lasso()
model4 = RandomForestRegressor()
model5 = KNeighborsRegressor()


model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
model4.fit(X_train,y_train)
model5.fit(X_train,y_train)

model1_rmse = np.sqrt(mean_squared_error(y_test,model1.predict(X_test)))
model2_rmse = np.sqrt(mean_squared_error(y_test,model2.predict(X_test)))
model3_rmse = np.sqrt(mean_squared_error(y_test,model3.predict(X_test)))
model4_rmse = np.sqrt(mean_squared_error(y_test,model4.predict(X_test)))
model5_rmse = np.sqrt(mean_squared_error(y_test,model5.predict(X_test)))
rmse = np.array([model1_rmse,model2_rmse,model3_rmse,model4_rmse,model5_rmse])


model1_mae = mean_absolute_error(y_test,model1.predict(X_test))
model2_mae = mean_absolute_error(y_test,model2.predict(X_test))
model3_mae = mean_absolute_error(y_test,model3.predict(X_test))
model4_mae = mean_absolute_error(y_test,model4.predict(X_test))
model5_mae = mean_absolute_error(y_test,model5.predict(X_test))
mae = np.array([model1_mae,model2_mae,model3_mae,model4_mae,model5_mae])


model1_r2 = r2_score(y_test,model1.predict(X_test))
model2_r2 = r2_score(y_test,model2.predict(X_test))
model3_r2 = r2_score(y_test,model3.predict(X_test))
model4_r2 = r2_score(y_test,model4.predict(X_test))
model5_r2 = r2_score(y_test,model5.predict(X_test))
r2 = np.array([model1_r2,model2_r2,model3_r2,model4_r2,model5_r2])

final_model = RandomForestRegressor().fit(X,y)

def error_res(feat):
    features = pd.DataFrame(columns = X.columns)
    features['Number of Generators'] = pd.DataFrame([feat[0]])
    features['Plant Capacity Factor'] = pd.DataFrame([feat[1]])
    features['Plant Generator Capacity (MW)'] = pd.DataFrame([feat[2]])
    features['Annual Heat Input (MMBTU)'] = pd.DataFrame([feat[3]])
    features['Annual Net Generation (MWh)'] = pd.DataFrame([feat[4]])
    features['Coal Percentage (%)'] = pd.DataFrame([feat[5]])
    features['Oil Percentage (%)'] = pd.DataFrame([feat[6]])
    features['Gas Percentage (%)'] = pd.DataFrame([feat[7]])
    features['Nuclear Percentage (%)'] = pd.DataFrame([feat[8]])
    features['Hydro Percentage (%)'] = pd.DataFrame([feat[9]])
    features['Biomass Percentage (%)'] = pd.DataFrame([feat[10]])
    features['Wind Percentage (%)'] = pd.DataFrame([feat[11]])
    features['Solar Percentage (%)'] = pd.DataFrame([feat[12]])
    features['Geothermal Percentage (%)'] = pd.DataFrame([feat[13]])
    features['Other Percentage (%)'] = pd.DataFrame([feat[14]])
    
    return final_model.predict(features)

def con1(feat):
    return 100 - abs(feat[5]) + abs(feat[6]) + abs(feat[7]) + abs(feat[8]) + abs(feat[9]) + abs(feat[10]) + abs(feat[11]) + abs(feat[12]) + abs(feat[13]) + abs(feat[14])

def con2(feat):
    return 770723- feat[4]

def con3(feat):
    tally = 0
    for ii in range(0,len(feat)):
        if feat[ii]<0:
            tally += 1
    
    return tally

x0 =[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]

cons_f1 = NonlinearConstraint(con1, 0 , 0)
cons_f2 = NonlinearConstraint(con2, 0 , 0)
cons_f3 = NonlinearConstraint(con3, 0 , 0)

bounds = [(0,10),(0,1),(0,1000),(0,5e6),(770723,770723),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100),(0,100)]

result = scp.optimize.differential_evolution(error_res,bounds, constraints= (cons_f1,cons_f2,cons_f3))
    



