import numpy as np
import torch
import torchsummary
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve


class AE_Trainer:
	def __init__(self, args):
		self.device = args.device
		self.epochs = args.ae_epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.ae_lr
		self.weight_decay = args.ae_weight_decay
		self.criterion = nn.MSELoss().to(self.device)
		self.train_loader = None
		self.test_loader = None
	
	def set_train_loader(self, loader):
		self.train_loader = loader

	def set_test_loader(self, loader):
		self.test_loader = loader


	def train(self, model):
		loader = self.train_loader
		model = model.to(self.device)

		total_batch = len(loader)
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

#		for epoch in tqdm(range(1,self.epochs+1)):
		for epoch in range(1,self.epochs+1):
			model.train()
			avg_cost =0
			
#			for batch_idx, (X,_, semi) in tqdm(enumerate(loader), total=total_batch):
			for batch_idx, (X,_, semi) in enumerate(loader):
				X = X.to(self.device)

				optimizer.zero_grad()
				hypothesis = model(X)
				
				scores= torch.sum((hypothesis - X)**2, dim=tuple(range(hypothesis.dim())))
				cost = torch.mean(scores)
				cost.backward()
				optimizer.step()
				avg_cost += cost/total_batch
			scheduler.step()

		torch.save(model.state_dict(), '../save_model/'+str(model.name)+'.pth')

		return model
	
	def test(self, model):
		loader = self.test_loader
		model = model.to(self.device)
		model.eval()
		with torch.no_grad():
			total_batch = len(loader)

			optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)

			avg_cost =0
			for batch_idx, (X,Y,semi) in tqdm(enumerate(loader), total=total_batch):
				X = X.to(self.device)

				hypothesis = model(X)
				
				cost = self.criterion(hypothesis, X)
				avg_cost += cost/total_batch
		return X.to('cpu').numpy(), hypothesis.to('cpu').numpy()

class CNN_Trainer:
	def __init__(self, args):
		self.device = args.device
		self.epochs = args.epochs
		self.batch_size = args.batch_size
		self.learning_rate = args.lr
		self.weight_decay = args.weight_decay
		self.test_interval = args.test_interval
		self.criterion = nn.MSELoss().to(self.device)
		self.train_loader = None
		self.test_loader = None
		self.eps = 1e-9
		self.eta = args.eta
	
	def set_train_loader(self, loader):
		self.train_loader = loader

	def set_test_loader(self, loader):
		self.test_loader = loader


	def train(self, ae_model, model):
		model = self.set_CNN(ae_model,model).to(self.device)


		self.c = self.set_c(model, self.train_loader)
		

		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
		total_batch = len(self.train_loader)

		auroc_list = []
		thr_list = []
		acc_list = []

#		for epoch in tqdm(range(1,self.epochs+1)):
		for epoch in range(1,self.epochs+1):
			model.train()
			avg_cost =0

			
#			for batch_idx, (X,Y,semi) in tqdm(enumerate(self.train_loader), total=total_batch):
			for batch_idx, (X,Y,semi) in enumerate(self.train_loader):
				X = X.to(self.device)
				Y = Y.to(self.device)
				semi = semi.to(self.device)

				optimizer.zero_grad()
				hypothesis = model(X)
				
				dist = torch.sum((hypothesis -self.c)**2, dim=1)
				cost = torch.where(semi==-1, self.eta*((dist + self.eps)**-1.0),dist).mean()

				cost.backward()
				optimizer.step()
				avg_cost += cost/total_batch

			scheduler.step()

			if (epoch+1) % self.test_interval == 0:
				auc, acc, thr = self.test(self.c, model)
				print("Epoch: {}/{} AUROC: {:.3f} ACC: {:.3f} thr: {}".format(epoch, self.epochs, auc, acc, thr))
				torch.save({"C": self.c, "model":model.state_dict()}, '../save_model/'+str(model.name)+'_'+str(epoch+1)+'.tar')
				auroc_list.append(auc)
				thr_list.append(thr)
				acc_list.append(acc)


		torch.save({"C": self.c, "model":model.state_dict()}, '../save_model/'+str(model.name)+'.tar')

		return self.c, auroc_list, acc_list, thr_list
	
	def test(self, C, model):	
		model = model.to(self.device)

		acc_sum =0
		score_list = []
		label_list = []

		model.eval()
		with torch.no_grad():
			total_batch = len(self.test_loader)


#			for batch_idx, (X,Y,_) in tqdm(enumerate(self.test_loader), total=total_batch):
			for batch_idx, (X,Y,_) in enumerate(self.test_loader):
				X = X.to(self.device)
				Y = Y.to(self.device)

				hypothesis = model(X)
					
				dist = torch.sum((hypothesis -C)**2, dim=1)
		
				score = dist

				score_list += score.to('cpu')
				label_list += Y.to('cpu')

		score_list = np.array(score_list)
		label_list = np.array(label_list)
		auc = roc_auc_score(label_list, score_list)
		_,_,self.thresholds = roc_curve(label_list, score_list)

		self.thr, acc = self.cal_thr(label_list, score_list, self.thresholds)
		return auc, acc, self.thr

		

	def cal_thr(self,labels, scores, thresholds):
		pre_acc = 0
		for t in thresholds:
			score = np.where(scores<t,0,1)
			t_acc = score == labels
			t_acc = np.sum(t_acc)/len(score)*100
			if t_acc > pre_acc:
				thr = t
				acc = t_acc
			pre_acc = t_acc
		return thr, acc


	def set_c(self, model, loader):
		c = torch.zeros(model.rep_dim, device=self.device)

		model.eval()
		with torch.no_grad():
			total_batch = len(loader)
			for batch_idx, (X, _, _) in enumerate(loader):
				X = X.to(self.device)

				hypothesis = model(X)
				c += torch.sum(hypothesis, dim=0).detach()
		c /= total_batch*self.batch_size

		c[(abs(c) < 0.1) & (c <0)] = -0.1
		c[(abs(c) < 0.1) & (c >0)] = 0.1

		return c
	
			
	def set_CNN(self, ae_model, model):
		ae = ae_model.state_dict()
		cnn = model.state_dict()

		ae = {k:v for k,v in ae.items() if k in cnn}

		cnn.update(ae)
		model.load_state_dict(cnn)
		return model


