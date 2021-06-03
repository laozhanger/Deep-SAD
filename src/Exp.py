import subprocess
import argparse
import openpyxl
from main import main 
import numpy as np

config = {
	'device': [str, 'cuda'],
	'batch_size': [int, 200],
	'ae_epochs': [int, 50],
	'ae_lr': [float, 1e-4],
	'ae_weight_decay': [float, 1e-6],

	'pre_train_model': [str, './model.pth'],
	'epochs': [int, 100],
	'lr': [float, 1e-5],
	'weight_decay': [float, 1e-6],
	'eta': [float, 1],

	'test_interval': [int, 10],

	'gamma_l': [float, 0],

	'pre_train': [bool, 'store_true'],
	'train': [bool, 'store_true'],
	'test': [bool, 'store_true'],
	'normal_class': [int, 5],
	'abnormal_class': [list, [1]],

	'test_model': [str, './MNIST_CNN.pth']
}

def set_args(config):
	p = argparse.ArgumentParser()

	for k,v in config.items():
		if type(v[0]) == bool:
			p.add_argument('--' + k, action=v[1])
		else:
			p.add_argument('--' + k, type=v[0], default=v[1])
	return p.parse_args()
	
def Exp():
	args = set_args(config)

	wb = openpyxl.Workbook()
	
	gamma_l = [0,0.05,0.1, 0.15, 0.2]
	ws = wb.create_sheet('Exp1',index=0)
	str_gamma = [str(g) for g in gamma_l]
	ws.append(["Normal","Abnormal"] + gamma_l)
	#Normal
	auc_list = []
	for i in range(10):
		#Abnormal
		abnormal = [index for index in range(10)]
		abnormal.remove(i)
		config['normal_class'][1]=i
		for j in range(9):
			config['abnormal_class'][1]=[abnormal[j]]
			g_auc = []
			for g in gamma_l:
				config['gamma_l'][1] = g
				print("Normal {} Abnormal {} Gamma_l {}".format(i, abnormal[j], g))
				args = set_args(config)
				auc,_ = main(args)
				g_auc.append(auc)
			ws.append([i,j] +g_auc)
			auc_list.append(g_auc)
	avg = np.mean(auc_list,0)
	avg = list(avg)
	ws.append(["",""]+avg)
	wb.save("./result.xlsx")
			

if __name__ == "__main__":
	Exp()

