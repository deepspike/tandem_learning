import torch
import torch.backends.cudnn as cudnn
import time
import os, sys

home_dir = os.getcwd()
sys.path.insert(0, home_dir)

from models.cifar10 import CifarNetIF
from data.data_loader_cifar10 import get_train_valid_loader, get_test_loader
from lib.classification import training, testing

# Load datasets
data_dir = os.path.join(home_dir, 'data/')
ckp_dir = os.path.join(home_dir, 'exp/cifar10/')

(train_loader, val_loader) = get_train_valid_loader(data_dir)
test_loader = get_test_loader(data_dir)

if __name__ == '__main__':
	if torch.cuda.is_available():
		device = 'cuda'
		print('GPU is available')
	else:
		device = 'cpu'
		print('GPU is not available')

	# Hyperparameters
	Tencode = 1  # time window size
	num_epochs = 150
	global best_acc_snn
	best_acc_snn = 0  # init with a low value
	neuronParam = {
		'neuronType': 'IF',
		'vthr': 1,
	}
	train_acc_history = []
	train_loss_history = []
	test_acc_history = []
	test_loss_history = []

	# Init Model and training configuration
	model = CifarNetIF(neuronParam, Tencode)
	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
	criterion = torch.nn.CrossEntropyLoss()

	if device == 'cuda':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True

	for epoch in range(num_epochs):
		since = time.time()

		# Training Stage
		model, acc_train, loss_train = training(model, train_loader, optimizer, criterion, device)

		# Validating stage
		acc_val, loss_val = testing(model, val_loader, criterion, device)

		# Testing Stage
		acc_test, loss_test = testing(model, test_loader, criterion, device)

		scheduler.step() # update learning rate

		train_acc_history.append(acc_train)
		train_loss_history.append(loss_train)
		test_acc_history.append(acc_test)
		test_loss_history.append(loss_test)

		# Training Progress Update
		time_elapsed = time.time() - since
		print('Epoch {:d} takes {:.0f}m {:.0f}s'.format(epoch + 1, time_elapsed // 60, time_elapsed % 60))
		print('Train Accuracy: {:4f}, Loss: {:4f}'.format(acc_train, loss_train))
		print('Validation Accuracy: {:4f}, Loss: {:4f}'.format(acc_val, loss_val))
		print('Test Accuracy: {:4f}'.format(acc_test))

		# Save Model
		if acc_test >= best_acc_snn:
			print("Saving the model.")

			if not os.path.isdir(ckp_dir):
				os.makedirs(ckp_dir)

			state = {
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'acc': acc_test,
			}
			torch.save(state, ckp_dir+'if_ckp.pt')
			best_acc_snn = acc_test

	best_test_acc = max(test_acc_history)
	print('Test Accuracy of best SNN model {}'.format(best_test_acc))
