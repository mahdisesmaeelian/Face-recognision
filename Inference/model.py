import tensorflow as tf
from tensorflow.keras.layers import  Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras import Model

width = height = 224

class MahdisNet(Model):
  def __init__(self):
    super().__init__()

    self.Conv2D_1 = Conv2D(32, (3,3), activation = 'relu',input_shape=(width, height, 3))
    self.Conv2D_2 = Conv2D(64, (3,3), activation = 'relu')
    self.MaxPooling = MaxPooling2D()
    self.flatten = Flatten()
    self.dense_1 = Dense(128, activation = 'relu')
    self.dense_2 = Dense(14, activation='softmax')
    self.dropout = Dropout(0.5)

  def call(self, x):
    y = self.Conv2D_1(x)
    z = self.MaxPooling(y)
    j = self.Conv2D_2(z)
    k = self.MaxPooling(j)
    m = self.flatten(k)
    n = self.dense_1(m)
    w = self.dropout(n)
    out = self.dense_2(w)

    return out 