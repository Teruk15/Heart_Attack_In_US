import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


###############################################################################################################
### Using Simple Linear Regression ###

df = pd.read_csv("./Heart_Attack_In_US/us_death.csv")
    # print(df.head())


### No missing value ###
sum_of_missing_value = df.isnull().sum()
    # print(sum_of_missing_value)



### Find the most values with correlation to death ###
index = df.corr().abs().nlargest(2,'DEATH').index
value = df.corr().abs().nlargest(2,'DEATH').values[0:len(df.index)-1,len(df.columns)-1]




### Store Smoke as x and Death as y
x_data = pd.DataFrame(df['SMOKE'],columns = ['SMOKE']).values
y_data = df['DEATH']




### Create train and test data
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.8,train_size=0.2,random_state=5)



### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train,y_train)




### Find equation ###
coef = model.coef_
intercept = model.intercept_
equ = 'Y = '+ str(intercept) + " + " + str(coef[0]) + '(X1)'
print(equ)



### Find R-squared vaue ###
print("R-squared: %.4f" % model.score(x_test,y_test))



###################################################################################################
### Turn data to polynomial ###

poly = PolynomialFeatures(degree = 3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)



### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train_poly,y_train)



### Find equation ###
coef_poly = model.coef_
intercept_poly = model.intercept_
equ_poly= 'Y = '+ str(intercept_poly) + " + " + str(coef_poly[1]) + '(X)' +  " + "  + str(coef_poly[2]) + '(X^2)' +  " + "  + str(coef_poly[3]) + '(X^3)'
print(equ_poly)



### Find R-squared vaue ###
print("R-squared: %.4f" % model.score(x_test_poly,y_test))


