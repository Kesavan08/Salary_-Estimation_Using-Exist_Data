from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import numpy as np

#read the csv
ds=pd.read_csv("salary.csv")

print(ds)
print(ds.shape)
print(ds.head(5))
print(ds.tail(5))

#segerigated to dataset
X=ds.iloc[:,:-1].values
print(X)

Y=ds.iloc[:,-1].values
print(Y)


#maping the data
#income=set(ds['income'])
#daataset['income']=dataset['income'].map({'<=50':0,'>50k':1}).astype(int)
#Spliting the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#Feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)

#model selection
error =[]
for i in range(1,40):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    pred_i=model.predict(X_test)
    error.append(np.mean(pred_i!=Y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title("Error Rate K value")
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()

#Training

model=KNeighborsClassifier(n_neighbors=20, metric='minkowski',p=2)
model.fit(X_train,Y_train)




#Predicting , whaeather new Customer with Age & Salary with Buy or NOT

age=int(input("Enter New Customer's Age:"))
edu=int(input("Enter New Customer's Salary:"))
                 
cg=int(input("Enter the captional"))
wh=int(input("Enter the work per/hour for week"))
newData=[[age,edu,cg,wh]]
res=model.predict(sc.transform(newData))

print(res)
if res==1:
    print("Customer will Buy")
else:
    print("Cutomer won't But")

#Prediction for all Test Data
y_pred=model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(Y_test),1)),1))


#check Acurancey
cm=confusion_matrix(Y_test,y_pred)
print("Confusion Matrix:")
print(cm)

print("Accurcy of the Model :{0}%".format(accuracy_score(Y_test,y_pred)*100))
