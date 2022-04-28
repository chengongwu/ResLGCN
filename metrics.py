from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import numpy as np

# 定义加权平均绝对百分比误差
def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
	# 两个矩阵都是n行276列
	total_sum=np.sum(Y_true)
	average=[]
	for i in range(len(Y_true)):
		for j in range(len(Y_true[0])):
			if Y_true[i][j]>0:
				# 加权
				temp=(Y_true[i][j]/total_sum)*np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
				average.append(temp)
	return np.sum(average)


# 定义评价函数
def evaluate_performance(Y_test_original,predictions):
	RMSE = sqrt(mean_squared_error(Y_test_original, predictions))
	print('RMSE is: '+str(RMSE))                                    # 均方根误差
	R2 = r2_score(Y_test_original, predictions)
	print("R2 is："+str(R2))                                        # 方差
	MAE=mean_absolute_error(Y_test_original, predictions)
	print("MAE is："+str(MAE))                                      # 平均绝对误差
	WMAPE=weighted_mean_absolute_percentage_error(Y_test_original, predictions)
	print("WMAPE is"+str(WMAPE))                                    # 加权平均绝对百分比误差
	return RMSE, R2, MAE, WMAPE