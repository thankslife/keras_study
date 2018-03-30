import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#Conv2D  图片是3通道 Conv1D  单声道或者双声道


# 先读入数据
(X_train, y_train), (X_test, y_test) = mnist.load_data("../test_data_home")
# 看一下数据集的样子
print(X_train[0].shape)#X_train是很多张图片[0]是第一张
print(y_train[0])

# 下面把训练集中的手写黑白字体变成标准的四维张量形式，即（样本数量，长，宽，1）
# 1是channel通道
# 并把像素值变成浮点格式
# X_train.shape[0]取得是行的数量
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 由于每个像素值都介于0到255，所以这里统一除以255，把像素值控制在0-1范围
X_train /= 255 #X_train是一个矩阵 这里相当于里面每个数都除以255
X_test /= 255


# 由于输入层需要10个节点，所以最好把目标数字0-9做成One Hot编码的形式
def tran_y(y):#自己定义的One_hot编码格式 先定义10个零 然后在对应个数上填1
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


# 把标签用One Hot编码重新表示一下
y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])#列表生成器
y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])

# 搭建卷积神经网络
model = Sequential()
# 添加一层卷积层，构造64个过滤器(卷积核)，每个过滤器(卷积核)覆盖范围是3*3*1
# 过滤器步长为1，图像四周补一圈0，并用relu进行非线性变化
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 input_shape=(28, 28, 1), activation='relu'))#activation如果不设置，就是线性的
# 添加一层最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))#默认Strides是和池化维度一样 这里是2*2 池化默认一般不做padding
# 设立Dropout层，Dropout的概率为0.5
model.add(Dropout(0.5))

# 重复构造，搭建深度网络
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# 把当前层节点展平
model.add(Flatten())

# 构造全连接层神经网络层
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))#如果是回归直接写None

# 定义损失函数，一般来说分类问题的损失函数都选择采用交叉熵
# adagrad自适应的调整学习率的随机梯度下降算法 因为自适应所以会比较耗时间
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# 放入批量样本，进行训练
model.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe)#验证集作用边训练边测试 在每一个step中都会验证
          , epochs=20, batch_size=128)
#epochs是迭代多少轮次 学完一轮是整个数据集/128 这里面有20轮

# 在测试集上评价模型的准确率
# verbose : 进度表示方式。0表示不显示数据，1表示显示进度条
scores = model.evaluate(X_test, y_test_ohe, verbose=0)


