#%%
import matplotlib.pyplot as plt
import segyio
import numpy as np
import scipy
import scipy.ndimage as snd

from tensorflow.keras.layers import Conv2D, InputLayer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# %%
segyfile = 'shots.segy'

f = segyio.open(segyfile,ignore_geometry=True)

#%%
marm = f.trace.raw[:]
shape = marm.shape
noise =  1.5 * marm.std() * np.random.random(marm.shape)
noisy = marm + noise

plt.figure(figsize=[20,20])
plt.subplot(1,2,1)
plt.imshow(marm,cmap=plt.cm.seismic)
plt.subplot(1,2,2)
plt.imshow(noisy,cmap=plt.cm.seismic)


#%%
def split_section(section,dX,dY,nchan=1):

    nx = int(np.floor(section.shape[0] / dX))
    ny = int(np.floor(section.shape[1] / dY))
    nimg = nx*ny

    scaler = MinMaxScaler()

    out = np.zeros((nimg,dX,dY,nchan))
    batch_pos=0
    for i in range(nx):
        for j in range(ny):

            x0= (i*dX)
            x1= (i*dX)+dX
            y0= (j*dY)
            y1= (j*dY)+dY

            temp = section[x0:x1,y0:y1]
            temp = scaler.fit_transform(temp)
            temp = temp.reshape(temp.shape[0],temp.shape[1],1)

            out[batch_pos] = temp
            batch_pos=batch_pos+1

    return out

y = split_section(marm,60,30)
X = split_section(noisy,60,30)


#%%
train_split = 0.2
last_sample = int(1-(len(X)*train_split))
X_train = X[:last_sample]
X_test = X[last_sample:]
y_train = y[:last_sample]
y_test = y[last_sample:]

batchsize=len(X_train)

#%%
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
model.compile(optimizer='rmsprop', loss='mse')


model.fit(x=X_train,y=y_train,batch_size=batchsize,epochs=100,verbose=0)


#%%
sample=0
pred= model.predict(X_test)[sample,:,:,0]
original = y_test[sample,:,:,0]
withnoise = X_test[sample,:,:,0]
dif = original-pred
mse = np.mean(dif*dif)
print(mse)

vmin=0
vmax=1
cmap = plt.cm.seismic
# cmap = 'viridis'
plt.figure(figsize=[20,20])
plt.subplot(1,4,1)
plt.imshow(original,cmap=cmap)#,vmin=vmin,vmax=vmax)
plt.title('seis')
plt.subplot(1,4,2)
plt.imshow(withnoise,cmap=cmap)#,vmin=vmin,vmax=vmax)
plt.title('noised')
plt.subplot(1,4,3)
plt.imshow(pred,cmap=cmap)#,vmin=vmin,vmax=vmax)
plt.title('pred')
plt.subplot(1,4,4)
plt.imshow(dif,cmap=cmap)#,vmin=vmin,vmax=vmax)
plt.title('diff')
plt.show()

# %%
