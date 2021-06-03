import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class Dataset(torch.utils.data.Dataset):
	def __init__(self, args,AE=True, train=True):
		super(Dataset, self).__init__()

		self.normal =args.normal_class
		self.abnormal = list(map(int, args.abnormal_class))
		self.novelty = [i for i in range(10)]
		self.novelty.remove(self.normal)

		if train == True:
			train_data = MNIST(root='./', train=True, download=True, transform=ToTensor())
			data, label = train_data.data, train_data.targets
		else:
			test_data = MNIST(root='./', train=False, download=True, transform=ToTensor())
			data, label = test_data.data, test_data.targets

		if train == True:
			# Add unlabeled normal data
			indices = np.where(np.isin(label, self.normal))
			train_data = data[indices]
			train_label = label[indices]

			# Add labeled abnormal data wrt gamma_l
			if AE==False:
				n = len(train_data)
				m = int(args.gamma_l/(1-args.gamma_l)*n)

				indices = np.where(np.isin(label, self.abnormal))[0]
				rand= np.random.permutation(len(indices))[:m]

				indices = indices[rand]
				data = torch.cat([train_data, data[indices]])
				label = torch.cat([train_label, label[indices]])
				semi= [0]*n + [-1]*m
			else:
				data = train_data
				label= train_label
				semi = [0] * len(data)
		else:
			semi = [0] * len(data)

		self.data = data/255.0
		self.data = self.data.unsqueeze(1)
		
		self.label = np.where(np.isin(label,self.normal),0,1)
		self.semi = semi
			
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self,index):
		return self.data[index], self.label[index], self.semi[index]





