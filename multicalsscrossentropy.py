from keras.models import Sequential
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
x,y=make_blobs(n_samples=100,centers=4,n_features=2,random_state=1)
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
dummy_y=np_utils.to_categorical(y)
print("dummy=",dummy_y.shape)

print("y",y)
print(x.shape)
print(y.shape)

def baselinemodel():
    model=Sequential()
    model.add(Dense(7,input_dim=2,activation="relu"))
    model.add(Dense(5,activation="relu"))
    model.add(Dense(4,activation="softmax"))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    return model
estimator=baselinemodel()

estimator.fit(x,dummy_y,epochs=200,shuffle=True)

xnew,yactual=make_blobs(n_samples=8,centers=4,n_features=2,random_state=1)
xnew=scaler.transform(xnew)
print("yactual",yactual)
dummy_y_actual=np_utils.to_categorical(yactual)
#ynew=model.predict_classes(xnew)
ynew = estimator.predict(xnew)
ynew = np.round(ynew).astype(int)

for i in range(len(ynew)):
    print(ynew[i],dummy_y_actual[i])

