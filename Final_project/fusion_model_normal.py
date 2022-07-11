import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader, Dataset
import random
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
from model import CNN1, CNN2, CNN3, CNN4, MLP, AE, EEGNet, SCCNet, ShallowConvNet
BATCH_SIZE = 128

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def choose_scheme(path, scheme, test_sub):
  filedir = os.listdir(path)
  x_train, x_test, y_train, y_test = [], [], [], []
  x_fine_tune, y_fine_tune = [], []
  # trial_class0, trial_class1, 
  # 將第一個subject 1固定作為testing subject
  subject_t = f"BCIC_S0{test_sub}_T.mat"
  subject_e = f"BCIC_S0{test_sub}_E.mat"
  x_train = torch.Tensor(io.loadmat(os.path.join(path, subject_t))['x_train']).unsqueeze(1)
  y_train = torch.Tensor(io.loadmat(os.path.join(path, subject_t))['y_train']).view(-1).long()
  x_test = torch.Tensor(io.loadmat(os.path.join(path, subject_e))['x_test']).unsqueeze(1)
  y_test = torch.Tensor(io.loadmat(os.path.join(path, subject_e))['y_test']).view(-1).long()
  len_x = x_train.size()[0]

  # 依序將其他subject的data讀入
  for filename in filedir:
    #print(filename)
    if filename in {subject_t, subject_e}:
      continue
    elif filename.endswith('E.mat'):
      x = torch.Tensor(io.loadmat(os.path.join(path, filename))['x_test']).unsqueeze(1)
      y = torch.Tensor(io.loadmat(os.path.join(path, filename))['y_test']).view(-1).long()
      x_train = torch.cat([x_train, x])
      y_train = torch.cat([y_train, y])
    elif filename.endswith('T.mat'):
      x = torch.Tensor(io.loadmat(os.path.join(path, filename))['x_train']).unsqueeze(1)
      y = torch.Tensor(io.loadmat(os.path.join(path, filename))['y_train']).view(-1).long()
      x_train = torch.cat([x_train, x])
      y_train = torch.cat([y_train, y])

  # choose real training and testing data based on scheme
  if scheme == 'individual': # indiviual will access only training session from subject1
    return [x_train[:len_x], y_train[:len_x], x_test, y_test]
  elif scheme == 'dependent': # dependent scheme collect all data except for the testing sessions from sub1
    return [x_train, y_train, x_test, y_test]
  elif scheme == 'independent': # independent scheme collect all data except for testing and training session from sub1
    return [x_train[len_x:], y_train[len_x:], x_test, y_test]
  elif scheme == 'fine-tune': # the data used on fine tune is the same as individual scheme
    return [x_train[:len_x], y_train[:len_x], x_test, y_test]
  else:
    raise ValueError('unexpected scheme, enter other scheme again')
  
def get_dataloader(data_path, scheme, test_sub):
  data = []
  data = choose_scheme(data_path, scheme, test_sub)

  x_train, y_train, x_test, y_test = data

  print("x_train shape: ", x_train.size())
  print("y_train shape: ", y_train.size())
  print("x_test shape: ", x_test.size())
  print("y_test shape: ", y_test.size())

  # 存成tensordataset
  train_dataset = TensorDataset(x_train, y_train)
  test_dataset = TensorDataset(x_test, y_test)
  # 包成dataloader
  train_dl = DataLoader(
      dataset = train_dataset,
      batch_size = BATCH_SIZE,
      shuffle = True,
      num_workers = 0   
  )
  test_dl = DataLoader(
      dataset = test_dataset,
      batch_size = len(test_dataset),
      shuffle = False,
      num_workers = 0   
  )
  return [train_dl, test_dl]


def val_model(model, test_dl, device):
  model['fusion model'].eval()
  # print('test start')
  acc = 0
  with torch.no_grad():
        for x_test, y_test in test_dl:
            x_test, y_test = x_test.to(device), y_test.to(device)
            latent1, _ = model['CNN1'](x_test) # model prediction
            latent2, _ = model['CNN2'](x_test)
            latent3, _ = model['CNN3'](x_test)
            latent4, _ = model['CNN4'](x_test)
            feat_vec = torch.cat([latent1, latent2, latent3, latent4], 1)
            _, pred= model['fusion model'](feat_vec)
            pred = torch.argmax(pred,1)
            acc += (pred==y_test).sum().item()

  acc /= len(test_dl.dataset)
 
  return acc

## if using MLP to fusion
def train_fusion_model(model, train_dl, test_dl, device, config):
    # set the optimizer and corresponding loss function
    optimizer = getattr(optim, config['optimizer'])(model["fusion model"].parameters(), lr=config['lr'], weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    model['CNN1'].eval()
    model['CNN2'].eval()
    model['CNN3'].eval()
    model['CNN4'].eval()

    record = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
        }
    
    max_acc = 0
    print('\n training start')
    # start training process
    for epoch in range(config['epoch']):
        model["fusion model"].train()
        train_loss = 0
        train_acc = 0
        loss = 0
        for x_train, y_train in train_dl:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                latent1, _ = model['CNN1'](x_train) # model prediction
                latent2, _ = model['CNN2'](x_train)
                latent3, _ = model['CNN3'](x_train)
                latent4, _ = model['CNN4'](x_train)
                feat_vec = torch.cat([latent1, latent2, latent3, latent4], 1)
            _, out = model['fusion model'](feat_vec)

            loss = criterion(out, y_train) # calculate loss
            pred = torch.argmax(out, axis = 1) # max out the prediction
            train_loss += loss.detach().cpu().item()
            train_acc += (pred==y_train).sum().item() # count the corrected classified casese

            loss.backward() # derive backpropagation
            optimizer.step() # update model
        
        del latent1, latent2, latent3, latent4, feat_vec, out

        # save the model which produce maximum validation accuracy
        val_acc = val_model(model, test_dl, device)
        
        ## free tensor dataset
        torch.cuda.empty_cache()

        if(val_acc>max_acc):
            max_acc = val_acc
            torch.save(model['fusion model'].state_dict(), f"{config['save_path']}sub{config['test sub']}_{config['model']}_{config['scheme']}_{config['lr']}_{BATCH_SIZE}_test16")

        record['train_loss'].append(train_loss/len(train_dl))
        record['train_acc'].append(train_acc/len(train_dl.dataset))
        record['val_acc'].append(val_acc)
          
        #lr_scheduler.step()
        if (epoch+1)%10 == 0:
            print(f'epoch: {epoch+1}, training acc: {train_acc/len(train_dl.dataset)} val_acc: {val_acc}')
    with open("MCNN_normal_log.txt", 'a') as f:
            f.writelines(f"{config['test sub']}\t{max_acc}\n")
    
    return record

def main(test_sub):
    device = get_device()
    set_seed(33)
    scheme = 'independent' # individual, dependent, independent, fine-tune
    test_sub = test_sub
    file_dir = 'BCI_data'
    dl = get_dataloader(file_dir, scheme, test_sub)
    train_dl, test_dl = dl
    config = {
        'epoch': 500,
        'optimizer': 'Adam',
        'lr': 0.0001,
        'scheme': scheme,
        'test sub': test_sub,
        'model': 'MCNN',
        'save_path': 'CNN_model/'
        }
    cnn1 = CNN1(9700).to(device)
    cnn1.load_state_dict(torch.load(f"CNN_model/normal/sub{test_sub}_CNN1_{scheme}_0.0001_128", map_location=torch.device(device)))
    cnn2 = CNN2(6500).to(device)
    cnn2.load_state_dict(torch.load(f"CNN_model/normal/sub{test_sub}_CNN2_{scheme}_0.0001_128", map_location=torch.device(device)))
    cnn3 = CNN3(2200).to(device)
    cnn3.load_state_dict(torch.load(f"CNN_model/normal/sub{test_sub}_CNN3_{scheme}_0.0001_128", map_location=torch.device(device)))
    cnn4 = CNN4(700).to(device)
    cnn4.load_state_dict(torch.load(f"CNN_model/normal/sub{test_sub}_CNN4_{scheme}_0.0001_128", map_location=torch.device(device)))
    fusion_model = MLP().to(device)
    model = {"CNN1": cnn1,
        "CNN2": cnn2,
        "CNN3": cnn3,
        "CNN4": cnn4,
        "fusion model":fusion_model}

    record = train_fusion_model(model, train_dl, test_dl, device, config)

if __name__ == '__main__':
    for s in range(1,10):
        main(s)