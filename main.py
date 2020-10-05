#Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
import xlrd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from scipy import stats
from scipy.stats import kurtosis, skew

#Step 2: Load the dataset
price_data=pd.read_excel(r'C:\Users\19802\Documents\UNCC MAFI program\Python Projects\Python Project_SP500_Tesla_10yearsReturn\python_tesla_sp500_10years_monthlyreturn.xlsx', index_col=0)
print(price_data)

#Step 3: Data Cleaning

    #check the data types
print(price_data.dtypes)

    #change column names
price_data=price_data.rename(columns={'^GSPC return':'S&P 500 Return'})
price_data=price_data.rename(columns={'TSLA return':'Tesla Return'})

    #check for missing values
print(price_data.isna().any())   # both True means missing values exist in both columns

    #drop missing values
price_data=price_data.dropna()

    #verify missing values are gone
print(price_data.isna().any())  # both False - verified

#Step 4: Data Exploration

    #create a statistical summary
print(price_data.describe()) #define x&y data, x=sp500, y=tesla
x = price_data['S&P 500 Return']
y = price_data['Tesla Return']

    #build a scatter plot and format the plot
plt.scatter(x,y, marker="d",color='grey',label="Monthly Return")
plt.title('S&P 500 VS. Tesla Monthly Return (2010-2020)')
plt.xlabel('S&P 500')
plt.ylabel('Tesla')
plt.legend()
plt.show()

    #observed the pattern from scatterplot, so calculate correlation
print(price_data.corr())

    #create a histogram to look at distribution
price_data.hist(grid=False, color="cadetblue")
plt.show()

#Step 5: Build the Model

    #define variables
X=price_data[['S&P 500 Return']]
Y=price_data[['Tesla Return']]

    #split datasets to training set and testing set
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=1)  #20% random data going to testing data

    #create a linear regression model with training set
regression_model=LinearRegression()
regression_model.fit(x_train, y_train)
coefficient=regression_model.coef_[0][0]
intercept = regression_model.intercept_[0]
print("The coefficient is {:.4}".format(coefficient))
print("The intercept is {:.4}".format(intercept))

    #calculate a predicted value for a new input
prediction=regression_model.predict([[555]])
predicted_value=prediction[0][0]
print('The predicted value is {:.5}'.format(predicted_value))

    #get y prediction with testing set
y_predict = regression_model.predict(x_test)
print(y_predict[:5])

#Step 6: Evaluate the Model

    #use statsmodel library to recreate the same model
X2=sm.add_constant(X)
model=sm.OLS(Y,X2)
results=model.fit()

    #statistical summary
print(results.summary()) #create a summary of the model
print(results.conf_int()) #We are 95% confident that the coefficent is between 0.6116 and 2.0324

    #hyphothesis testing
results.pvalues
print(results.pvalues) #pvalue<0.05, reject null, there is relationship

    #fit the model
    #calculate mse, mae, rmse, r^2
model_mse=mean_squared_error(y_test, y_predict)
model_mae=mean_absolute_error(y_test,y_predict)
model_rmse=math.sqrt(model_mse)
model_r2=r2_score(y_test, y_predict)
print('MSE {:.4}'.format(model_mse))   #MSE: 0.01632
print('MAE {:.4}'.format(model_mae))   #MAE: 0.1038
print('RMSE {:.4}'.format(model_rmse)) #RMSE: 0.1277
print('R2 {:.4}'.format(model_r2))     #R2=-0.1203

    #plot the residuals (to check if they are normally distributed)
residual=y_test-y_predict
plt.hist(residual, color='royalblue')
plt.title("Model Residual Distribution")
plt.show()

    #plot the regression line
plt.scatter(x_test,y_test, color='gainsboro', label='price')
plt.plot(x_test,y_predict, color="green", linewidth=2, linestyle='-', label='Regression Line')
plt.title("Linear Regression Model S&P500 VS. Tesla Monthly Return (2010-2020)")
plt.xlabel("S&P500")
plt.ylabel("Tesla")
plt.legend()
plt.show()

#Step 7: Save the Model for the Future Use
import pickle

    #pickle the model
with open('my_linear_regression.sav','wb') as f:
    pickle.dump(regression_model,f)

    #load it back
with open('my_linear_regression.sav','rb') as f:
    regression_model_2 = pickle.load(f)

    #make a new prediction
print(regression_model_2.predict([[303]]))

