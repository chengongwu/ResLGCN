import time
import numpy as np
from LoadData import Get_All_Data
from ResLGCN import build_model

global_start_time = time.time()
def Save_Data(path,model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE,Run_epoch):
	print(Run_epoch)
	RMSE_ALL=[]
	R2_ALL=[]
	MAE_ALL=[]
	WMAPE_ALL=[]
	Average_train_time=[]
	RMSE_ALL.append(RMSE)
	R2_ALL.append(R2)
	MAE_ALL.append(MAE)
	WMAPE_ALL.append(WMAPE)
	model.save(path+str(Run_epoch)+'-model-with-graph.h5')
	np.savetxt(path+str(Run_epoch)+'-RMSE_ALL.txt', RMSE_ALL)
	np.savetxt(path+str(Run_epoch)+'-R2_ALL.txt', R2_ALL)
	np.savetxt(path+str(Run_epoch)+'-MAE_ALL.txt', MAE_ALL)
	np.savetxt(path+str(Run_epoch)+'-WMAPE_ALL.txt', WMAPE_ALL)
	with open(path+str(Run_epoch)+'-predictions.csv', 'w') as file:
		predictions = predictions.tolist()
		for i in range(len(predictions)):
			file.write(str(predictions[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	with open(path+str(Run_epoch)+'-Y_test_original.csv', 'w') as file:
		Y_test_original = Y_test_original.tolist()
		for i in range(len(Y_test_original)):
			file.write(str(Y_test_original[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	duration_time = time.time() - global_start_time
	Average_train_time.append(duration_time)
	np.savetxt(path+str(Run_epoch)+'-Average_train_time.txt', Average_train_time)
	print('total training time(s):', duration_time)

X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2,X_train_3,X_test_3,X_train_4,X_test_4=\
	Get_All_Data(TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)
# 初始训练epoch，以后每次加10，运行15次
Run_epoch = 10
for i in range(15):
	model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE = build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
		Y_test_original,batch_size=64,epochs=Run_epoch,a=a,time_lag=6)
	Save_Data("E:/OneDrive/桌面/文件/ResLSTM/TestResult/", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch)
	Run_epoch += 10


#For Get_All_Data, change parameters referring to this: TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360
#10min:10,6,108,5,540,eopch=200
#15min:15,6,72,5,360 eopch=140
#30min:30,6,36,5,180 eopch=200
#60min:60,6,18,5,90 eopch=235