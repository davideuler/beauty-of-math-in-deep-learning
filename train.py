import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

mnist = tf.keras.datasets.mnist
#载入 MNIST 数据集，并将整型转换为浮点型，除以 255 是为了归一化。
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#使用 tf.keras.Sequential 建立模型，并且选择优化器和损失函数
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#训练模型
history = model.fit(x_train, y_train, epochs=5,validation_data=(x_test,y_test))


#模型评估
model.evaluate(x_test,  y_test, verbose=2)

#查看训练集与测试集的均方误差和准确率变化情况
history.history.keys()

#查看 training set, validation set 的损失和准确率
plt.plot(history.epoch,history.history.get('loss'),label='Loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.epoch,history.history.get('accuracy'),label='Accuracy')
plt.plot(history.epoch,history.history.get('val_accuracy'),label='Validation Accuracy')
plt.legend()
plt.show()

# 保存全模型
model.save('tf_model.h5')