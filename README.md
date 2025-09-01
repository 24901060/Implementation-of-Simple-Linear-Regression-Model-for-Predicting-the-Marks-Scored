# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Organize the dataset into a structured format, such as a CSV file or a DataFrame, for easy handling and analysis.

2. Apply a Simple Linear Regression model to fit the training data and identify the relationship between the independent and dependent variables.

3. Use the trained model to make predictions on the test dataset and assess its generalization capability.

4. Evaluate the model's performance using standard regression metrics, including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), to measure prediction accuracy.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAVEEN  S
RegisterNumber:  212224230206
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
```
<img width="347" height="295" alt="image" src="https://github.com/user-attachments/assets/8cca6a35-4d7b-4d4a-8234-3e54833f5171" />

```
df.tail()
```

<img width="349" height="316" alt="image" src="https://github.com/user-attachments/assets/a2d1c547-81e9-4692-a7c5-cb9f36264c7b" />

```

x = df.iloc[:,:-1].values
x
```

<img width="880" height="620" alt="image" src="https://github.com/user-attachments/assets/13a36283-3bee-40a9-bd2f-759b37bb2710" />

```

y = df.iloc[:,1].values
y
```

<img width="1260" height="119" alt="image" src="https://github.com/user-attachments/assets/fc9b3078-495e-4a90-9b3b-10fc7f680980" />

```

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
```
```

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
```

```
y_pred
```

<img width="1116" height="140" alt="image" src="https://github.com/user-attachments/assets/210e657a-a2e6-4f4d-a7aa-ef5cbc1e167a" />

```

y_test
```

<img width="833" height="79" alt="image" src="https://github.com/user-attachments/assets/3f520467-3ae4-43a7-837e-b5eded8f24af" />

```

mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```

<img width="973" height="194" alt="image" src="https://github.com/user-attachments/assets/0b3a9090-5dc0-4a93-83bd-c6a169b91887" />

```

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

<img width="1139" height="732" alt="image" src="https://github.com/user-attachments/assets/eae9633f-4524-4258-aabe-a74340ac1b69" />

```

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

<img width="1291" height="682" alt="image" src="https://github.com/user-attachments/assets/f94d08d1-bc14-42fe-951a-d2be8beb44ad" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
