import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

###############################################################################################################
### Using Multiple Linear Regression ###

df = pd.read_csv("./Heart_Attack_In_US/us_death.csv")



### No missing value ###
sum_of_missing_value = df.isnull().sum()



### Find all values correlation ###
index = df.corr().abs().nlargest(6,'DEATH').index
value = df.corr().abs().nlargest(6,'DEATH').values[0:len(df.index)-1,len(df.columns)-1]
print(index)
print(value)
    # Index(['DEATH', 'SMOKE', 'PHY_IA', 'DIAB', 'HIGH_CH', 'OBES'], dtype='object')
    #[1.         0.60174703 0.55980962 0.45905533 0.44647791 0.43089113]



### Store Values x and Death as y
x_data = pd.DataFrame(np.c_[df['SMOKE'],df['PHY_IA'],df['DIAB'],df['HIGH_CH'],df['OBES']],columns = ['SMOKE','PHY_IA','DIAB','HIGH_CH','OBES']).values
y_data = df['DEATH']



### Create train and test data
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.70,train_size=0.3,random_state=5)



### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train,y_train)



### Find equation ###
coef = model.coef_
intercept = model.intercept_



### Find R-squared vaue ###
print("R-squared(linear): %.4f" % model.score(x_test,y_test))



### Real value vs Predicted value ###
# real_death = y_test
# pred_death = model.predict(x_test)
# plt.scatter(real_death,pred_death)
# plt.xlabel('Real Death')
# plt.ylabel('Predicted Death')
#plt.show()



############################################################################
### Using Multiple Polynomial Regression ###

deg = 2
poly_feature = PolynomialFeatures(degree = deg)
x_train_poly = poly_feature.fit_transform(x_train)
x_test_poly = poly_feature.fit_transform(x_test)



### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train_poly,y_train)



### Find R-squared vaue ###
print("R-squared(polynomial): %.4f" % model.score(x_test_poly,y_test))