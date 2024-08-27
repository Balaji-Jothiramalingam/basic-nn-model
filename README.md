# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural Network regression model is a type of machine learning algorithm inspired by the structure of the brain. It excels at identifying complex patterns within data and using those patterns to predict continuous numerical values.his includes cleaning, normalizing, and splitting your data into training and testing sets. The training set is used to teach the model, and the testing set evaluates its accuracy. This means choosing the number of layers, the number of neurons within each layer, and the type of activation functions to use.The model is fed the training data.Once trained, you use the testing set to see how well the model generalizes to new, unseen data. This often involves metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).Based on the evaluation, you might fine-tune the model's architecture, change optimization techniques, or gather more data to improve its performance.

## Neural Network Model


![360812287-4c944a47-f65f-422d-b335-f6ed484ed8b1](https://github.com/user-attachments/assets/667493bf-19f7-402b-b54b-49e51762e46b)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:BALAJI J
### Register Number:212221243001
```python
```
## Importing Required package
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
````
## Authenticate the Google sheet
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet=gc.open('Untitled spreadsheet').sheet1
data=worksheet.get_all_values()

dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'Input':float})
dataset1=dataset1.astype({'Output':float})
dataset1.head()
X=dataset1[['Input']].values
y=dataset1[['Output']].values

```
## Split the testing and training data
```
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=33)
Scaler=MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)

```
## Build the Deep learning Model
```
ai_brain=Sequential([
     Dense(units=8,activation='relu'),
     Dense(units=10,activation='relu'),
     Dense(1)
])

ai_brain.compile(optimizer='adam',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

````
## Evaluate the Model

```


loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[4]]
X_n1=Scaler.transform(X_n1)
ai_brain.predict(X_n1)





```
## Dataset Information

![307531202-cdef71ea-4774-4bf2-baf2-9d7dae7d9592](https://github.com/user-attachments/assets/fad99bc0-7e21-4153-a1bf-dd2d2af44bdb)


## OUTPUT

### Training Loss Vs Iteration Plot

![307531317-a7b48087-1179-4781-8786-e3d160344202](https://github.com/user-attachments/assets/f63b19d9-36fa-46fa-85bb-83f27bc46acf)


## Epoch training

![307531367-1247ecf7-80e4-4443-ab84-c09d0cd4d541](https://github.com/user-attachments/assets/d6788a70-0173-4132-ae17-f06147954f6c)


### Test Data Root Mean Squared Error


![307531378-0114d30a-8081-4205-a158-95efe5450804](https://github.com/user-attachments/assets/3f69a782-1d1b-420f-9983-ba8d54d21245)

### New Sample Data Prediction


![307531392-ea52cc7b-b09f-400c-90e8-a8170793c2ef](https://github.com/user-attachments/assets/71ad103d-0d61-4b48-8cab-d01fddb2c7b2)


## RESULT

Thus a basic neural network regression model for the given dataset is written and executed successfully.
