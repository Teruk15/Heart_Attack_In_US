import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Heart_Attack_In_US/death.csv")


### No missing value ###
sum_of_missing_value = df.isnull().sum()



### Find two values correlates most ###
index = df.corr().abs().nlargest(3,'DEATH').index
value = df.corr().abs().nlargest(3,'DEATH').values[0:len(df.index)-1,len(df.columns)-1]

    # ['DEATH', 'SMOKE', 'OBES']
    # [1. 0.50062738 0.42024417]
    # Shows Smoke and Obesity have two biggest correlation to number of Death



### Plot Smoke and Obesity to Dath ###
# plt.scatter(df['OBES'],df['DEATH'],marker='o')
# plt.xlabel('Obesity')
# plt.ylabel('Death')
    #plt.show()

# plt.scatter(df['SMOKE'],df['DEATH'],marker='o')
# plt.xlabel('Smoking')
# plt.ylabel('Death')
    #plt.show()



### Store Obesity and Smoke as x and Death as y
x_data = pd.DataFrame(np.c_[df['OBES'],df['SMOKE']],columns = ['OBES','SMOKE'])
y_data = df['DEATH']




### Create train and test data
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.7,train_size=0.3,random_state=5)


### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train.values,y_train)




### Real value vs Predicted value ###
real_death = y_test
pred_death = model.predict(x_test.values)
plt.scatter(real_death,pred_death)
plt.xlabel('Real Death')
plt.ylabel('Predicted Death')
#plt.show()



### Find equation ###
coef = model.coef_
intercept = model.intercept_
equ = 'Y = '+ str(intercept) + " + " + str(coef[0]) + '(X1)' + " + " +  str(coef[1]) + '(X2)'
print(equ)



### Find R-squared vaue ###
print("R-squared: %.4f" % model.score(x_test.values,y_test))



### Predict data using equation ###
obesity1,smoke1 = 35,22
obesity2,smoke2 = 40,30
obesity3,smoke3 = 30,20

print(model.predict([[obesity1,smoke1]]))
print(model.predict([[obesity2,smoke2]]))
print(model.predict([[obesity3,smoke3]]))


###################################################################
#Improve data


### Setup polynmial function ###
degree = 2
polynomial_features = PolynomialFeatures(degree = degree)



### Linear x_train ---> Polynomial x_train ###
x_train_poly = polynomial_features.fit_transform(x_train.values)    
    # print(polynomial_features.get_feature_names_out(['x1','x2']))
    # ['1' 'x1' 'x2' 'x1^2' 'x1 x2' 'x2^2']



### Linear x_test ---> Polynomial x_test ###
x_test_poly = polynomial_features.fit_transform(x_test.values)
    # print(polynomial_features.get_feature_names_out(['x1','x2']))
    # ['1' 'x1' 'x2' 'x1^2' 'x1 x2' 'x2^2']



### Create module ###
model = LinearRegression()



### Train model ###
model.fit(x_train_poly,y_train)



### Find R-squared vaue ###
print("R-squared: %.4f" % model.score(x_test_poly,y_test))