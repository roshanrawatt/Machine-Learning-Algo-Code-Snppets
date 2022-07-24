from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
x,y=make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

model=Sequential()
model.add(Dense(4,input_dim=2,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.fit(x,y,epochs=500)

xnew,yactual=make_blobs(n_samples=3,centers=2,n_features=2,random_state=1)
xnew=scaler.transform(xnew)
print("yactual",yactual)
#ynew=model.predict_classes(xnew)
ynew = model.predict(xnew)
ynew = np.round(ynew).astype(int)

for i in range(len(ynew)):
    print(xnew[i],ynew[i],yactual[i])