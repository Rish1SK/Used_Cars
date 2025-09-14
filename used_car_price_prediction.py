#Importing required packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
matplotlib.style.use('ggplot')
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
np.random.seed(7)
#Reading DataSet
df=pd.read_csv("usedcardata.csv")
del df["L"] #Deleting unnecessary column
df.set_index("Name",inplace=True) #Setting Name as Index
df.isnull().sum() #Checking for null values

#Splitting Data Set into test and train
dfy=df["Price"]
dfx=df.drop("Price",axis=1)
x_train, x_test, y_train, y_test = train_test_split(dfx,dfy)

#Visualizing the relationship between Power and Price of car
df.plot(kind="scatter",x="Power",y="Price",figsize=(9,9))
 

#Converting categorical varliable Transmission to numeric
train_trans=pd.get_dummies(x_train["Transmission"])
x_train=pd.concat([x_train,train_trans],axis=1)
del x_train["Transmission"]

#Doing the same for test dataset
test_trans=pd.get_dummies(x_test["Transmission"])
x_test=pd.concat([x_test,test_trans],axis=1)
del x_test["Transmission"]

#Converting categorical varliable Fuel_type to numeric
train_Fuel=pd.get_dummies(x_train["Fuel_Type"])
x_train=pd.concat([x_train,train_Fuel],axis=1)
del x_train["Fuel_Type"]

#Doing the same for test dataset
test_Fuel=pd.get_dummies(x_test["Fuel_Type"])
x_test=pd.concat([x_test,test_Fuel],axis=1)
del x_test["Fuel_Type"]

#Converting categorical varliable Owner_type to numeric
train_ow=pd.get_dummies(x_train["Owner_Type"])
x_train=pd.concat([x_train,train_ow],axis=1)
del x_train["Owner_Type"]
x_train = x_train.rename(columns={'Fourth & Above':'Fourth'})

#Doing the same for test dataset
test_ow=pd.get_dummies(x_test["Owner_Type"])
x_test=pd.concat([x_test,test_ow],axis=1)
del x_test["Owner_Type"]
x_test = x_test.rename(columns={'Fourth & Above':'Fourth'}) 

#Converting categorical varliable Location to numeric
train_loc=pd.get_dummies(x_train["Location"])
x_train=pd.concat([x_train,train_loc],axis=1)
del x_train["Location"]

#Doing the same for test dataset
test_loc=pd.get_dummies(x_test["Location"])
x_test=pd.concat([x_test,test_loc],axis=1)
del x_test["Location"]


print(x_train.head())
print(x_test.head())
#Calculating multiple linear regression
mlr=linear_model.LinearRegression()
mlr.fit(x_train,y_train)
train_pred=mlr.predict(x_train)
test_pred = mlr.predict(x_test)
r2 = r2_score(y_test, test_pred)
print("R^2 score on test data:", r2)

st.set_page_config(page_title="Used Car Price Prediction",page_icon=":car")
st.title("Car Price Predictor")
locs=["Ahmedabad","Bangalore","Chennai","Coimbatore","Delhi","Hyderabad","Jaipur","Kochi","Kolkata","Mumbai","Pune"]
ft=["CNG","Diesel","LPG","Petrol"]
tr=["Automatic","Manual"]
ot=["First","Second","Third","Fourth"]

col1, col2, col3 = st.columns(3)

with col1:
  Location=st.selectbox("Select your location",options=locs)
  Year=st.number_input("Enter the year your car was bought")
  Owner_Type=st.selectbox("Select your ownership type",options=ot)
  seats=st.number_input("Enter the number of seats in your car")


with col2:
  Fuel_Type=st.selectbox("Select your car fuel type",options=ft)
  Transmission=st.selectbox("Select your car transmission type",options=tr)
  engine=st.number_input("Enter your car engine capacity")


with col3:
  power=st.number_input("Enter your car horse power")
  kl=st.number_input("Enter the kilometers run by the car")
  mileage=st.number_input("Enter your car mileage")

pdf=pd.DataFrame()
pdf.insert(0,"Year",[Year])
pdf.insert(1,"Kilometers_Driven",[kl])
pdf.insert(2,"Mileage",[mileage])
pdf.insert(3,"Engine",[engine])
pdf.insert(4,"Power",[power])
pdf.insert(5,"Seats",[seats])
if Transmission=="Automatic":
  pdf.insert(6,"Automatic",1)
else:
  pdf.insert(6,"Automatic",0)

if Transmission=="Manual":
  pdf.insert(7,"Manual",1)
else:
  pdf.insert(7,"Manual",0)

if Fuel_Type=="CNG":
  pdf.insert(8,"CNG",1)
else:
  pdf.insert(8,"CNG",0)

if Fuel_Type=="Diesel":
  pdf.insert(9,"Diesel",1)
else:
  pdf.insert(9,"Diesel",0)

if Fuel_Type=="LPG":
  pdf.insert(10,"LPG",1)
else:
  pdf.insert(10,"LPG",0)

if Fuel_Type=="Petrol":
  pdf.insert(11,"Petrol",1)
else:
  pdf.insert(11,"Petrol",0)

if Owner_Type=="First":
  pdf.insert(12,"First",1)
else:
  pdf.insert(12,"First",0)


if Owner_Type=="Fourth":
  pdf.insert(13,"Fourth",1)
else:
  pdf.insert(13,"Fourth",0)


if Owner_Type=="Second":
  pdf.insert(14,"Second",1)
else:
  pdf.insert(14,"Second",0)


if Owner_Type=="Third":
  pdf.insert(15,"Third",1)
else:
  pdf.insert(15,"Third",0)

if Location=="Ahmedabad":
  pdf.insert(16,"Ahmedabad",1)
else:
  pdf.insert(16,"Ahmedabad",0)

if Location=="Bangalore":
  pdf.insert(17,"Bangalore",1)
else:
  pdf.insert(17,"Bangalore",0)

if Location=="Chennai":
  pdf.insert(18,"Chennai",1)
else:
  pdf.insert(18,"Chennai",0)

if Location=="Coimbatore":
  pdf.insert(19,"Coimbatore",1)
else:
  pdf.insert(19,"Coimbatore",0)

if Location=="Delhi":
  pdf.insert(20,"Delhi",1)
else:
  pdf.insert(20,"Delhi",0)

if Location=="Hyderabad":
  pdf.insert(21,"Hyderabad",1)
else:
  pdf.insert(21,"Hyderabad",0)

if Location=="Jaipur":
  pdf.insert(22,"Jaipur",1)
else:
  pdf.insert(22,"Jaipur",0)

if Location=="Kochi":
  pdf.insert(23,"Kochi",1)
else:
  pdf.insert(23,"Kochi",0)

if Location=="Kolkata":
  pdf.insert(24,"Kolkata",1)
else:
  pdf.insert(24,"Kolkata",0)

if Location=="Mumbai":
  pdf.insert(25,"Mumbai",1)
else:
  pdf.insert(25,"Mumbai",0)

if Location=="Pune":
  pdf.insert(26,"Pune",1)
else:
  pdf.insert(26,"Pune",0)

test=mlr.predict(pdf)
st.sidebar.header("Welcome to the Car Price Predictor")
st.sidebar.write("Your go-to destination for estimating the value of your second-hand car. Whether you're looking to buy, sell, or simply curious about your car's worth, our user-friendly website provides you with a quick and accurate assessment based on the details you provide.")
st.sidebar.subheader("How It Works")
st.sidebar.markdown(
"""
- Enter basic information like when and where the car was bought
- Enter car specifications like horsepower, engine capacity, fuel type, transmission and seats
- Enter how much kilometers the car has been driven and the mileage its giving
- Click the Predict button then our algorithm will analyze these details along with current market trends and generate an estimated price for your car.
"""
)
res=st.button("Predict")
x=test[0]
x=round(x,2)
x=str(x)
L=" Lakhs"
x=x+L
if res:
 st.subheader("The price of your car is")
 st.write(x)
