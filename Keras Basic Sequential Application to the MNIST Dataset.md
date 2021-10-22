### **Goal:**
Test out a quick artificial neural network model for the MNIST dataset using a sequential model to construct a neural network that can classify the handwritten digits for the MNIST dataset. Technically for a neural network to be considered an implementation of Deep Learning there has to be roughly 3 or more node layers, so this application uses two sequential models; one that is considered Deep Learning and one that is not.


```python
# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# deep learning imports
import keras
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical # makes one-hot encoding easy
from keras.callbacks import EarlyStopping

# notebook settings
%matplotlib inline
pd.options.display.max_columns = 100
```

The data could have been loaded directly through keras. I chose not to do this as it is important to remember that the fit function here requires a one-hot encoded target set which the "to_categorical" function helps to achieve.


```python
# load the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])




```python
# set X and target matrices
X, y = mnist['data'], mnist['target'].astype('int')

target = to_categorical(y)
target
```




    array([[0., 0., 0., ..., 0., 0., 0.],
           [1., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)



Like when training any other model, we use a validation set to determine how well the model will perform on unseen data. With keras, that option is built in via the "validation_split" parameter. The trouble here is that there is not a way to stratify the validation sample which could affect model performance if there is a noticeable imbalance in the data. The following checks to see if there is an imbalance prior to running the model: 


```python
pd.Series(y).value_counts(normalize=True)
```




    1    0.112529
    7    0.104186
    3    0.102014
    2    0.099857
    9    0.099400
    0    0.098614
    6    0.098229
    8    0.097500
    4    0.097486
    5    0.090186
    dtype: float64



The proportions are roughly the same so stratified samples are not necessary.

Typically Deep Learning is applied when large amounts of training data are available, however it performs surprisingly well when trained on only 20% of the data and predicting on the other 80% as shown below:


```python
# attempt to train on 20% of the daa and then predict on the rest

# build the model
model = Sequential()

# add early stopping 
early_stopping_monitor = EarlyStopping(patience=3)

# add the layers - rough guess to start
model.add(Dense(50, activation='relu', input_shape=(len(mnist['feature_names']),))) #minimum is zero so reLU would work nicely here
model.add(Dense(50, activation='relu'))

# add final layer
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit and validate
model.fit(X, target, validation_split=.80, epochs=30, callbacks=[early_stopping_monitor])


```

    Epoch 1/30
    438/438 [==============================] - 2s 4ms/step - loss: 4.0522 - accuracy: 0.6783 - val_loss: 1.0826 - val_accuracy: 0.7819
    Epoch 2/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.7459 - accuracy: 0.8286 - val_loss: 0.7163 - val_accuracy: 0.8392
    Epoch 3/30
    438/438 [==============================] - 1s 2ms/step - loss: 0.4999 - accuracy: 0.8717 - val_loss: 0.6722 - val_accuracy: 0.8590
    Epoch 4/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.4343 - accuracy: 0.8898 - val_loss: 0.5861 - val_accuracy: 0.8728
    Epoch 5/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.3864 - accuracy: 0.8996 - val_loss: 0.6141 - val_accuracy: 0.8665
    Epoch 6/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.3470 - accuracy: 0.9100 - val_loss: 0.5392 - val_accuracy: 0.8885
    Epoch 7/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.3387 - accuracy: 0.9149 - val_loss: 0.5091 - val_accuracy: 0.8859
    Epoch 8/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.3071 - accuracy: 0.9206 - val_loss: 0.5500 - val_accuracy: 0.8964
    Epoch 9/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2875 - accuracy: 0.9264 - val_loss: 0.4927 - val_accuracy: 0.8952
    Epoch 10/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2649 - accuracy: 0.9287 - val_loss: 0.4965 - val_accuracy: 0.9021
    Epoch 11/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2499 - accuracy: 0.9359 - val_loss: 0.4517 - val_accuracy: 0.9078
    Epoch 12/30
    438/438 [==============================] - 1s 2ms/step - loss: 0.2457 - accuracy: 0.9373 - val_loss: 0.4484 - val_accuracy: 0.9119
    Epoch 13/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2333 - accuracy: 0.9389 - val_loss: 0.4938 - val_accuracy: 0.9063
    Epoch 14/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2147 - accuracy: 0.9423 - val_loss: 0.4618 - val_accuracy: 0.9108
    Epoch 15/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2148 - accuracy: 0.9439 - val_loss: 0.4789 - val_accuracy: 0.9107





    <keras.callbacks.History at 0x1584ba580>



The model shows a validation accuracy of 91% for a 20/80 split of the data which is really good for a basic model trained on only 20% of the data. The following model shows the improvement from using a more proper split of the data (75/25).


```python
# attempt to train on 20% of the data and then predict on the rest

# build the model
model = Sequential()

# add early stopping 
early_stopping_monitor = EarlyStopping(patience=3)

# add the layers - rough guess to start
model.add(Dense(50, activation='relu', input_shape=(len(mnist['feature_names']),))) #minimum is zero so reLU would work nicely here
model.add(Dense(50, activation='relu'))

# add final layer
model.add(Dense(10, activation='softmax'))

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit and validate
model.fit(X, target, validation_split=.25, epochs=30, callbacks=[early_stopping_monitor])

```

    Epoch 1/30
    1641/1641 [==============================] - 3s 1ms/step - loss: 1.6087 - accuracy: 0.8008 - val_loss: 0.4826 - val_accuracy: 0.8755
    Epoch 2/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.4124 - accuracy: 0.8920 - val_loss: 0.3701 - val_accuracy: 0.9003
    Epoch 3/30
    1641/1641 [==============================] - 2s 996us/step - loss: 0.3226 - accuracy: 0.9147 - val_loss: 0.3094 - val_accuracy: 0.9249
    Epoch 4/30
    1641/1641 [==============================] - 2s 947us/step - loss: 0.2738 - accuracy: 0.9275 - val_loss: 0.2628 - val_accuracy: 0.9325
    Epoch 5/30
    1641/1641 [==============================] - 2s 939us/step - loss: 0.2375 - accuracy: 0.9363 - val_loss: 0.2737 - val_accuracy: 0.9302
    Epoch 6/30
    1641/1641 [==============================] - 2s 947us/step - loss: 0.2073 - accuracy: 0.9426 - val_loss: 0.2189 - val_accuracy: 0.9439
    Epoch 7/30
    1641/1641 [==============================] - 2s 939us/step - loss: 0.1852 - accuracy: 0.9471 - val_loss: 0.1959 - val_accuracy: 0.9470
    Epoch 8/30
    1641/1641 [==============================] - 2s 962us/step - loss: 0.1640 - accuracy: 0.9539 - val_loss: 0.2099 - val_accuracy: 0.9470
    Epoch 9/30
    1641/1641 [==============================] - 2s 956us/step - loss: 0.1566 - accuracy: 0.9548 - val_loss: 0.2261 - val_accuracy: 0.9403
    Epoch 10/30
    1641/1641 [==============================] - 2s 943us/step - loss: 0.1447 - accuracy: 0.9573 - val_loss: 0.1803 - val_accuracy: 0.9519
    Epoch 11/30
    1641/1641 [==============================] - 2s 949us/step - loss: 0.1371 - accuracy: 0.9602 - val_loss: 0.1784 - val_accuracy: 0.9554
    Epoch 12/30
    1641/1641 [==============================] - 2s 943us/step - loss: 0.1286 - accuracy: 0.9630 - val_loss: 0.1771 - val_accuracy: 0.9580
    Epoch 13/30
    1641/1641 [==============================] - 2s 943us/step - loss: 0.1181 - accuracy: 0.9659 - val_loss: 0.1699 - val_accuracy: 0.9579
    Epoch 14/30
    1641/1641 [==============================] - 2s 941us/step - loss: 0.1171 - accuracy: 0.9658 - val_loss: 0.1897 - val_accuracy: 0.9527
    Epoch 15/30
    1641/1641 [==============================] - 2s 941us/step - loss: 0.1107 - accuracy: 0.9677 - val_loss: 0.1809 - val_accuracy: 0.9565
    Epoch 16/30
    1641/1641 [==============================] - 2s 954us/step - loss: 0.1080 - accuracy: 0.9683 - val_loss: 0.1938 - val_accuracy: 0.9535





    <keras.callbacks.History at 0x15859e550>



The model shows a validation accuracy of 95% for a 75/25 split of the data which is amazing for a basic model trained on only 20% of the data.

While the previous is a neural network, there are not enough layers for it to be considered a Deep Learning model. The following takes the model and increase the layers and nodes per layer to see if this enhances the accuracy of the model.


```python
# build the model
model2 = Sequential()

# add the layers - rough guess to start
model2.add(Dense(75, activation='relu', input_shape=(len(mnist['feature_names']),))) #minimum is zero so reLU would work nicely here
model2.add(Dense(75, activation='relu'))
model2.add(Dense(75, activation='relu'))
model2.add(Dense(75, activation='relu'))

# add final layer
model2.add(Dense(10, activation='softmax'))

# compile
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit and validate
model2.fit(X, target, validation_split=.80, epochs=30, callbacks=[early_stopping_monitor])

```

    Epoch 1/30
    438/438 [==============================] - 3s 6ms/step - loss: 1.8992 - accuracy: 0.7495 - val_loss: 0.5820 - val_accuracy: 0.8463
    Epoch 2/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.4515 - accuracy: 0.8826 - val_loss: 0.4788 - val_accuracy: 0.8703
    Epoch 3/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2898 - accuracy: 0.9204 - val_loss: 0.3651 - val_accuracy: 0.9042
    Epoch 4/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.2283 - accuracy: 0.9338 - val_loss: 0.3638 - val_accuracy: 0.9074
    Epoch 5/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1960 - accuracy: 0.9449 - val_loss: 0.3315 - val_accuracy: 0.9210
    Epoch 6/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1760 - accuracy: 0.9491 - val_loss: 0.3146 - val_accuracy: 0.9246
    Epoch 7/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1677 - accuracy: 0.9524 - val_loss: 0.3561 - val_accuracy: 0.9152
    Epoch 8/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1326 - accuracy: 0.9599 - val_loss: 0.3325 - val_accuracy: 0.9245
    Epoch 9/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1302 - accuracy: 0.9614 - val_loss: 0.3080 - val_accuracy: 0.9281
    Epoch 10/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1290 - accuracy: 0.9617 - val_loss: 0.3315 - val_accuracy: 0.9302
    Epoch 11/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1156 - accuracy: 0.9671 - val_loss: 0.3399 - val_accuracy: 0.9272
    Epoch 12/30
    438/438 [==============================] - 1s 3ms/step - loss: 0.1336 - accuracy: 0.9636 - val_loss: 0.3279 - val_accuracy: 0.9288





    <keras.callbacks.History at 0x1586cb6a0>




```python
# build the model
model2 = Sequential()

# add the layers - rough guess to start
model2.add(Dense(75, activation='relu', input_shape=(len(mnist['feature_names']),))) #minimum is zero so reLU would work nicely here
model2.add(Dense(75, activation='relu'))
model2.add(Dense(75, activation='relu'))
model2.add(Dense(75, activation='relu'))

# add final layer
model2.add(Dense(10, activation='softmax'))

# compile
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit and validate
model2.fit(X, target, validation_split=.25, epochs=30, callbacks=[early_stopping_monitor])

```

    Epoch 1/30
    1641/1641 [==============================] - 3s 1ms/step - loss: 0.9397 - accuracy: 0.8392 - val_loss: 0.2792 - val_accuracy: 0.9207
    Epoch 2/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.2502 - accuracy: 0.9294 - val_loss: 0.2162 - val_accuracy: 0.9377
    Epoch 3/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.1915 - accuracy: 0.9462 - val_loss: 0.1939 - val_accuracy: 0.9449
    Epoch 4/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.1657 - accuracy: 0.9527 - val_loss: 0.1580 - val_accuracy: 0.9543
    Epoch 5/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.1402 - accuracy: 0.9598 - val_loss: 0.1639 - val_accuracy: 0.9551
    Epoch 6/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.1246 - accuracy: 0.9638 - val_loss: 0.1512 - val_accuracy: 0.9583
    Epoch 7/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.1147 - accuracy: 0.9672 - val_loss: 0.1441 - val_accuracy: 0.9610
    Epoch 8/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0992 - accuracy: 0.9715 - val_loss: 0.1457 - val_accuracy: 0.9629
    Epoch 9/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0908 - accuracy: 0.9741 - val_loss: 0.1347 - val_accuracy: 0.9657
    Epoch 10/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0823 - accuracy: 0.9762 - val_loss: 0.1264 - val_accuracy: 0.9686
    Epoch 11/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0776 - accuracy: 0.9783 - val_loss: 0.1556 - val_accuracy: 0.9629
    Epoch 12/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0753 - accuracy: 0.9790 - val_loss: 0.1366 - val_accuracy: 0.9682
    Epoch 13/30
    1641/1641 [==============================] - 2s 1ms/step - loss: 0.0672 - accuracy: 0.9814 - val_loss: 0.1320 - val_accuracy: 0.9704





    <keras.callbacks.History at 0x1589743a0>



The new model while only increasing to 92% when training on 20% of the data, increased to 97% accuracy when training on a proper split (75/25).


```python

```
