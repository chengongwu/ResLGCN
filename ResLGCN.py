import numpy as np
import os
import keras
from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model                  # 可视化模型
from keras.models import load_model
from keras.optimizers import Adam
from metrics import evaluate_performance

np.set_printoptions(threshold=np.inf)                         # 设置打印时的输出方式
keras.backend.set_image_data_format('channels_last')          # 返回默认图像的维度顺序

os.environ["PATH"] += os.pathsep + 'E:\OneDrive\桌面\ResLSTM'  # 将模型输出可视化

# 残差块结构设计
def Unit(x, filters, pool=False):
	res = x
	if pool:                                                                                    # 如果进行池化操作
		x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)                                   # 将输入进行最大池化操作，池化卷积核大小（2,2），使用same填充
		res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)  # 进行卷积操作，卷积核大小（1,1）
	out = BatchNormalization()(x)                                                               # 将输入数据进行归一化输出
	out = Activation("relu")(out) # 对输出进行激活函数激活
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out) # 进行卷积操作，卷积核大小（3,3）

	out = BatchNormalization()(out) # 将输出数据进行归一化输出
	out = Activation("relu")(out)   # 对输出进行激活函数激活
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out) # 进行卷积操作，卷积核大小（3,3）

	out = keras.layers.add([res, out]) # 残差块输出

	return out

# LSTM注意力机制运用
def attention_3d_block(inputs,timesteps):
    a = Permute((2, 1))(inputs)                        # 进行维度转置
    a = Dense(timesteps, activation='linear')(a)       # 进行线性全连接操作
    a_probs = Permute((2, 1))(a)                       # 进行维度转置
    output_attention_mul = multiply([inputs, a_probs]) # 矩阵乘法
    return output_attention_mul

# 定义模型
def multi_input_model(time_lag):  # 构建多输入模型

    input1_ = Input(shape=(276, time_lag-1, 3), name='input1')  # 第一模块输入格式
    input2_ = Input(shape=(276, time_lag-1, 3), name='input2')  # 第二模块输入格式
    input3_ = Input(shape=(276, time_lag-1, 1), name='input3')  # 第三模块输入格式
    input4_ = Input(shape=(11, time_lag-1, 1), name='input4')   # 第四模块输入格式
    # 第一模块输入操作
    x1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input1_) # 进行卷积操作
	# 共2个残差块
    x1 = Unit(x1, 32)            # 第一个残差块操作
    x1 = Unit(x1, 64, pool=True) # 第二个残差块操作
    x1 = Flatten()(x1)           # 将多维数据一维化传入全连接层
    x1 = Dense(276)(x1)          # 进入全连接层，全连接操作

    # 第二模块输入操作
    x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input2_) # 进行卷积操作
    x2 = Unit(x2, 32)            # 第一个残差块操作
    x2 = Unit(x2, 64, pool=True) # 第二个残差块操作
    x2 = Flatten()(x2)           # 将多维数据一维化传入全连接层
    x2 = Dense(276)(x2)          # 进入全连接层，全连接操作

    # 第三模块输入操作
    x3 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input3_) # 进行卷积操作
    x3 = Unit(x3, 32)            # 第一个残差块操作
    x3 = Unit(x3, 64, pool=True) # 第二个残差块操作
    x3 = Flatten()(x3)           # 将多维数据一维化传入全连接层
    x3 = Dense(276)(x3)          # 进入全连接层，全连接操作

    # 第四模块输入操作
    x4 = Flatten()(input4_)                                          # 将多维数据一维化传入全连接层
    x4 = Dense(276)(x4)                                              # 进入全连接层，全连接操作
    x4 = Reshape(target_shape=(276, 1))(x4)                          # 改变数据形状输入LSTM
    x4 = LSTM(128, return_sequences=True, input_shape=(276, 1))(x4)  # 进入长短时记忆神经网络，返回全部hidden state值
    x4 = LSTM(276, return_sequences=False)(x4)                       # 进入长短时记忆神经网络，返回单个hidden state值
    x4 = Dense(276)(x4)                                              # 进入全连接层，全连接操作

    out = keras.layers.add([x1, x2, x3, x4])                         # 将四个模块输出数据进行合并作为下一阶段输入数据
    out = Reshape(target_shape=(276, 1))(out)                        # 格式化输入数据
    out = LSTM(128, return_sequences=True,input_shape=(276, 1))(out) # 进入长短时记忆神经网络，返回全部hidden state值
    out = attention_3d_block(out, 276)                               # 引入LSTM注意力机制，输出格式（276，128）
    out = Flatten()(out)                                             # 将多维数据一维化传入全连接层
    out = Dense(276)(out)                                            # 进入全连接层，全连接操作

    model = Model(inputs=[input1_, input2_, input3_,input4_], outputs=[out]) # 使用inputs与outputs建立函数链式模型
    return model  # 返回多输入模型

# 模型的训练与测试
def build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
	Y_test_original,batch_size,epochs,a,time_lag):
# 训练数据的格式化
	X_train_1 = X_train_1.reshape(X_train_1.shape[0],  276, time_lag-1, 3)
	X_train_2 = X_train_2.reshape(X_train_2.shape[0],  276, time_lag-1, 3)
	X_train_3 = X_train_3.reshape(X_train_3.shape[0],  276, time_lag-1, 1)
	X_train_4 = X_train_4.reshape(X_train_4.shape[0],  11, time_lag-1, 1)
	Y_train = Y_train.reshape(Y_train.shape[0], 276)
# 测试数据的格式化
	X_test_1 = X_test_1.reshape(X_test_1.shape[0],  276, time_lag-1, 3)
	X_test_2 = X_test_2.reshape(X_test_2.shape[0],  276, time_lag-1, 3)
	X_test_3 = X_test_3.reshape(X_test_3.shape[0],  276, time_lag-1, 1)
	X_test_4 = X_test_4.reshape(X_test_4.shape[0],  11, time_lag-1, 1)
	Y_test = Y_test.reshape(Y_test.shape[0], 276)

	if epochs == 50:  #训练轮数
		model = multi_input_model(time_lag) # 加载训练模型
		model.compile(optimizer=Adam(), loss='mse', metrics=['mse']) # Keras训练模型之前,需要配置学习过程，参数为：优化器、损失函数、评估标准
		# Adam优化是一种基于随机估计的一阶和二阶矩的随机梯度下降方法。
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)
		'''
		训练模型fit:为模型训练固定的epochs（数据集上的迭代）
		x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
        y：标签，numpy array
        batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
        epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
        verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
        注：validation_split=0.05
		'''
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)  # 预测值的输出
	else:
		# 每训练10次加载一下模型
		model = load_model('TestResult/'+str(epochs-10)+'_model_graph.h5')
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False)# , validation_split=0.05
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)

	#将输出进行反归一化
	predictions = np.zeros((output.shape[0], output.shape[1]))
	for i in range(len(predictions)):
		for j in range(len(predictions[0])):
			predictions[i, j] = round(output[i, j]*a, 0)
			if predictions[i, j] < 0:
				predictions[i, j] = 0

	RMSE,R2,MAE,WMAPE=evaluate_performance(Y_test_original,predictions)
	# 可视化模型结构
	plot_model(model, to_file='Design/model.png', show_shapes=True)
	return model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE # 返回模型以及各类误差值



