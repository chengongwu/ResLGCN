# 检测 tensorflow 能使用的设备情况
# from tensorflow.python.client import device_lib
import tensorflow as tf
# gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus=tf.config.experimental.list_physical_devices(device_type='CPU')
# #os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 这个可以指定使用哪个设备
# print(device_lib.list_local_devices())
# print(gpus, cpus)
print(tf.test.is_built_with_cuda())