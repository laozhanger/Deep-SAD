import numpy as np
import torch
import argparse
from argparse import RawTextHelpFormatter

from datasets import Dataset
from model.Trainer import *
from model.MNIST import *

parser = argparse.ArgumentParser(description="2020 Deep-SAD code reproduced by Jungi Lee", formatter_class=RawTextHelpFormatter)

#Model params
parser.add_argument('--device', type=str, default='cuda', help="Device name for torch environment")
parser.add_argument('--batch_size', type=int, default=128, help="Batch size")

#Pretrain params
parser.add_argument('--ae_epochs', type=int, default=150, help="Autoencoder num of epochs")
parser.add_argument('--ae_lr', type=float, default=1e-4, help="Autoencoder learning rate")
parser.add_argument('--ae_weight_decay', type=float, default=5e-3, help="Autoencoder weight decay")

#Train params
parser.add_argument('--pre_train_model', type=str, default='./model.pth', help="Pretrained model path")
parser.add_argument('--epochs', type=int, default=150, help="Model num of epochs")
parser.add_argument('--lr', type=float, default=1e-4, help="Model learning rate")
parser.add_argument('--weight_decay', type=float, default=5e-3, help="Model weight decay")
parser.add_argument('--eta', type=float, default=1, help="Params which determine influence of labeled anomalies")

parser.add_argument('--test_interval', type=int, default=10, help="log interval when training")

parser.add_argument('--gamma_l', type=float, default=0, help="Set proportion of labeled anomalies")

#Flag params
parser.add_argument('--pre_train', default=False, help="On Autoencoder(run pretrain)")
parser.add_argument('--train', action='store_true', default=True, help="training start Flag")
parser.add_argument('--test', action='store_true', default=True, help="Test start Flag")

#Preprocessing params
parser.add_argument('--normal_class', type=int, default=0, help="Select normal class(default: 0)")
parser.add_argument('--abnormal_class', nargs='*', default=[1], help="Select normal class(default: 0)")

parser.add_argument('--test_model', type=str, default='./MNIST_CNN.pth', help="Pretrained model path")
args = parser.parse_args()

def main(args):
	Train_dset = Dataset(args, AE=False,train=True)
	Test_dset = Dataset(args, AE=False, train=False)

	train_loader = torch.utils.data.DataLoader(dataset=Train_dset, batch_size=args.batch_size, shuffle=True, drop_last=False)
	test_loader = torch.utils.data.DataLoader(dataset=Test_dset, batch_size=args.batch_size, shuffle=False, drop_last=False)

	#AutoEncoder
	AE_model = MNIST_AE()
	if args.pre_train == True:
		AE_trainer = AE_Trainer(args)
		AE_trainer.set_train_loader(train_loader)

		AE_trainer.train(AE_model)
	else:
		AE_model.load_state_dict(torch.load(args.pre_train_model))

	#CNN
	Train_dset = Dataset(args, AE=False,train=True)
	Test_dset = Dataset(args, AE=False, train=False)

	train_loader = torch.utils.data.DataLoader(dataset=Train_dset, batch_size=args.batch_size, shuffle=True, drop_last=False)
	test_loader = torch.utils.data.DataLoader(dataset=Test_dset, batch_size=args.batch_size, shuffle=False, drop_last=False)

	CNN_model = MNIST_CNN()
	CNN_trainer = CNN_Trainer(args)
	CNN_trainer.set_train_loader(train_loader)
	CNN_trainer.set_test_loader(test_loader)

	if args.train==True:
		C, auc_list, acc_list, _ = CNN_trainer.train(AE_model, CNN_model)

		auc_index = np.argmax(auc_list)
		model_index = (auc_index+1)*args.test_interval

		if args.test==True:
			model_dict = torch.load("../save_model/"+str(CNN_model.name) + "_" + str(model_index) +".tar")
			C = model_dict['C']
			CNN_model.load_state_dict(model_dict['model'])
			auc, acc, thr = CNN_trainer.test(C,CNN_model)

	if args.test==True:
		if args.train == True:
			model_dict = torch.load("../save_model/"+str(CNN_model.name) + "_" + str(model_index) +".tar")
		else:
			model_dict = torch.load(args.test_model)
		C = model_dict['C']
		CNN_model.load_state_dict(model_dict['model'])
		auc, acc, thr = CNN_trainer.test(C,CNN_model)
	return auc, acc




if __name__ == "__main__":
	main(args)
