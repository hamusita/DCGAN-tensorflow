import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.models import *
from keras.layers import *
from keras.utils import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import umap
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# CIFAR10をロード
(X_train,y_train),(X_test, y_test) = cifar10.load_data()

print('X_train:', X_train.shape, 'y_train:', y_train.shape)
print('X_test:', X_test.shape, 'y_test:', y_test.shape)

l = list(zip(X_train, y_train))
np.random.shuffle(l)
X_train, y_train = zip(*l)

# データを図示
plt.figure(figsize=(20,20))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(str(y_train[i]),fontsize=14)
    plt.savefig('./cifar10.png', bbox_inches='tight')
plt.show()

## 正規化
X_train = np.array(X_train, dtype='float32')/255.
X_test = X_test/255.

## One-hot化
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

input1 = Input((32,32,3,))

conv1 = Conv2D(32, (2,2), padding='same', name='conv2D_1', kernel_initializer='he_normal')(input1)
conv1 = Conv2D(32, (2,2), padding='same', name='conv2D_2', kernel_initializer='he_normal')(conv1)
acti1 = Activation('relu', name='acti1')(conv1)
pool1 = MaxPool2D(pool_size=(2,2), name='pool1')(acti1)
drop1 = Dropout(0.2, name='drop1')(pool1)

conv2 = Conv2D(32, (2,2), padding='same', name='conv2D_3', kernel_initializer='he_normal')(drop1)
conv2 = Conv2D(32, (2,2), padding='same', name='conv2D_4', kernel_initializer='he_normal')(conv2)
acti2 = Activation('relu', name='acti2')(conv2)
pool2 = MaxPool2D(pool_size=(2,2), name='pool2')(acti2)
drop2 = Dropout(0.2, name='drop2')(pool2)

flat1 = Flatten(name='flat1')(drop2)


#### 出力を得たい層 ####
dens1 = Dense(512, name='hidden')(flat1)

acti3 = Activation('relu', name='acti3')(dens1)
dens2 = Dense(10,activation='softmax', name='end')(acti3)

model = Model(inputs=input1, outputs=dens2)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

history=model.fit(X_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.2)

predict_class = model.predict(X_test)
predict_class = np.argmax(predict_class, axis=1)
true_class = np.argmax(y_test, axis=1)
cmx = confusion_matrix(true_class, predict_class)

plt.figure(figsize=(12,12))
sns.heatmap(cmx, annot=True)
# plt.savefig('./cnn_predict.png', bbox_inches='tight')
plt.show()
print("Accuracy: {0}".format(accuracy_score(true_class, predict_class)))

hidden = hidden_model.predict(X_train)
hid_co = umap.UMAP().fit(hidden)
plt.scatter(hid_co.embedding_[:,0],
            hid_co.embedding_[:,1],
            c = np.argmax(y_train, axis=1),
            cmap='plasma')
plt.colorbar()
plt.savefig('./train_umap.png', bbox_inches='tight')
plt.show()


