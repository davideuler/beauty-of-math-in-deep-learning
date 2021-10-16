import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

#调用模型
new_model = tf.keras.models.load_model('tf_model.h5')

#调用模型进行预测识别
filename = "digit-number-3.jpg"
im = Image.open(filename)  #读取图片路径
im = im.resize((28,28)) #调整大小和模型输入大小一致
im = np.array(im)

#对图片进行灰度化处理
p3 = im.min(axis = -1)
plt.imshow(p3,cmap = 'gray')
plt.show()

#将白底黑字变成黑底白字   由于训练模型是这种格式
for i in range(28):
   for j in range(28):
       p3[i][j] = 255-p3[i][j]

#模型输出结果是每个类别的概率，取最大的概率的类别就是预测的结果
ret = new_model.predict((p3/255).reshape((1,28,28)))
number = np.argmax(ret) 
print("filename: %s predicted:%s" % (filename, number) )
