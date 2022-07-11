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
BATCH_SIZE = 8
STRIDE = 16

writer = SummaryWriter('log/')

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## for orignial dataset
def window_slide(x, y):
  length = x.size()[3]
  new_length = int(length/2)
  t1 = x[:, :, :, 0:new_length]
  t2 = x[:, :, :, int(length/4):int(length/4)+new_length]
  t3 = x[:, :, :, int(length/2):int(length/2)+new_length]
  x = torch.stack((t1, t2, t3), 1).view(x.size(0)*3,1,22,new_length)
  y = torch.stack((y,y,y), 1).view(864)
  return x, y

def sliding_window(x,y):
    stride = 16
    new_x, new_y = [],[]
    new_x = [x[:,:,i*stride:i*stride+288] for i in range(int(288/stride)+1)]
    new_x = torch.stack(new_x,1).view(x.size(0)*(int(288/stride)+1),1,22,288)
    new_y = [y for _ in range(int(288/stride)+1)]
    new_y = torch.stack(new_y,1).view(y.size()*(int(288/stride)+1))
    return new_x, new_y

class sliding_dataloader(Dataset):
    def __init__(self, data, label, stride):
        self.data = data
        self.label = label
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        x_new = [x[:, :, i*self.stride:i*self.stride+281].unsqueeze(0) for i in range(int(281/self.stride)+1)]
        x_new = torch.cat(x_new, 0)
        y_new = y.repeat(int(281/self.stride)+1)

        return x_new, y_new

def choose_scheme(path, scheme, test_sub):
  filedir = os.listdir(path)
  x_train, x_test, y_train, y_test, x_validation, y_validation = [], [], [], [], [], []
  x_fine_tune, y_fine_tune = [], []
  # 將第一個subject 1固定作為testing subject
  subject_t = f"BCIC_S0{test_sub}_T.mat"
  subject_e = f"BCIC_S0{test_sub}_E.mat"
  x_train = torch.Tensor(io.loadmat(os.path.join(path, subject_t))['x_train']).unsqueeze(1)
  y_train = torch.Tensor(io.loadmat(os.path.join(path, subject_t))['y_train']).view(-1).long()
  x_test = torch.Tensor(io.loadmat(os.path.join(path, subject_e))['x_test']).unsqueeze(1)
  y_test = torch.Tensor(io.loadmat(os.path.join(path, subject_e))['y_test']).view(-1).long()
  #x_train, y_train = window_slide(x_train, y_train)
  #x_test, y_test = window_slide(x_test, y_test)
  x_validation = x_test
  y_validation = y_test
  len_x = x_train.size()[0]

  # 依序將其他subject的data讀入
  for filename in filedir:
    #print(filename)
    if filename in {subject_t, subject_e}:
      continue
    elif filename.endswith('E.mat'):
      x = torch.Tensor(io.loadmat(os.path.join(path, filename))['x_test']).unsqueeze(1)
      y = torch.Tensor(io.loadmat(os.path.join(path, filename))['y_test']).view(-1).long()
    #   x, y = window_slide(x, y)
      x_train = torch.cat([x_train, x])
      y_train = torch.cat([y_train, y])
      
    elif filename.endswith('T.mat'):
      x = torch.Tensor(io.loadmat(os.path.join(path, filename))['x_train']).unsqueeze(1)
      y = torch.Tensor(io.loadmat(os.path.join(path, filename))['y_train']).view(-1).long()
    #   x, y = window_slide(x, y)
      x_train = torch.cat([x_train, x])
      y_train = torch.cat([y_train, y])
      

  # choose real training and testing data based on scheme
  if scheme == 'individual': # indiviual will access only training session from subject1
    return [x_train[:len_x], y_train[:len_x], x_test, y_test, x_validation, y_validation]
  elif scheme == 'dependent': # dependent scheme collect all data except for the testing sessions from sub1
    return [x_train, y_train, x_test, y_test, x_validation, y_validation]
  elif scheme == 'independent': # independent scheme collect all data except for testing and training session from sub1
    return [x_train[len_x:], y_train[len_x:], x_test, y_test, x_validation, y_validation]
  elif scheme == 'fine-tune': # the data used on fine tune is the same as individual scheme
    return [x_train[:len_x], y_train[:len_x], x_test, y_test, x_validation, y_validation]
  else:
    raise ValueError('unexpected scheme, enter other scheme again')
  
def get_dataloader(data_path, scheme, test_sub):
  data = []
  data = choose_scheme(data_path, scheme, test_sub)

  x_train, y_train, x_test, y_test, x_validation, y_validation = data
  x_train, y_train = x_train.to('cuda'), y_train.to('cuda')
  x_test, y_test = x_test.to('cuda'), y_test.to('cuda')

  print("x_train shape: ", x_train.size())
  print("y_train shape: ", y_train.size())
  print("x_validation shape: ", x_validation.size())
  print("y_validation shape: ", y_validation.size())
  print("x_test shape: ", x_test.size())
  print("y_test shape: ", y_test.size())


  # 存成tensordataset
  train_dataset = sliding_dataloader(x_train, y_train, 16)
  test_dataset = sliding_dataloader(x_test, y_test, 16)
  validation_dataset = sliding_dataloader(x_validation, y_validation, 16)
  # 包成dataloader
  train_dl = DataLoader(
      dataset = train_dataset,
      batch_size = BATCH_SIZE,
      shuffle = True,
      num_workers = 0   
  )
  test_dl = DataLoader(
      dataset = test_dataset,
      batch_size = 1,
      shuffle = False,
      num_workers = 0   
  )
  validation_dl = DataLoader(
      dataset = validation_dataset,
      batch_size = BATCH_SIZE,
      shuffle = False,
      num_workers = 0   
  )
  return [train_dl, test_dl, validation_dl]

def train_model(model, train_dl, test_dl, device, config):
  # set the optimizer and corresponding loss function
  optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'], weight_decay=0.001)
  criterion = nn.CrossEntropyLoss()
  #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
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
    model.train()
    train_loss = 0
    train_acc = 0
    for x_train, y_train in train_dl:
      x_train, y_train = torch.flatten(x_train, 0,1).to(device), torch.flatten(y_train, 0,1).to(device)
      optimizer.zero_grad()
      _, out = model(x_train) # model prediction
      loss = criterion(out, y_train) # calculate loss
      pred = torch.argmax(out, axis = 1) # max out the prediction
      train_loss += loss.detach().cpu().item()
      train_acc += (pred==y_train).sum().item() # count the corrected classified casese

      loss.backward() # derive backpropagation
      optimizer.step() # update model
    
    # save the model which produce maximum validation accuracy
    val_acc = val_model(model, test_dl, device)
    if(val_acc>max_acc):
      max_acc = val_acc
      torch.save(model.state_dict(), f"{config['save_path']}sub{config['subject']}_{config['model']}_{config['scheme']}_{config['lr']}_{BATCH_SIZE}_test16")
    writer.add_scalar('Train/loss', train_loss/len(train_dl), epoch)
    writer.add_scalar('Train/acc', train_acc/(len(train_dl.dataset)*y_train.size(0)), epoch)
    writer.add_scalar('Val/acc', val_acc, epoch)
    record['train_loss'].append(train_loss/len(train_dl))
    record['train_acc'].append(train_acc/(len(train_dl.dataset)*y_train.size(0)))
    record['val_acc'].append(val_acc)
      
    #lr_scheduler.step()
    if (epoch+1)%10 == 0:
      print(f'epoch: {epoch+1}, training acc: {train_acc/(len(train_dl.dataset)*(281//STRIDE+1))} val_acc: {val_acc}')
  
  return record

def val_model(model, test_dl, device):
  model.eval()
  # print('test start')
  acc = 0
  with torch.no_grad():
    for x_test, y_test in test_dl:
      x_test, y_test = x_test.squeeze(0).to(device), y_test.squeeze(0).to(device)
      _, pred = model(x_test)
      pred = pred.mean(0)
      pred = torch.argmax(pred)
      acc += (pred==y_test[0]).item()
  acc /= len(test_dl.dataset)
 
  return acc

def test_model(model, test_dl, device):
  model.eval()
  # print('test start')
  acc = 0
  latent = []
  output = []
  with torch.no_grad():
    for x_test, y_test in test_dl:
      x_test, y_test = x_test.to(device), y_test.to(device)
      vector, pred = model(x_test)
      pred = torch.argmax(pred, axis = 1)
      output.append(pred)
      latent.append(vector)
      acc += (pred==y_test).sum().item()
  acc /= len(test_dl.dataset)
  print(f"testing accuracy: {acc*100}%")
 
  return latent, output

def main(test_sub, i):
  device = get_device()
  set_seed(33)
  file_dir = 'BCI_data/'
  scheme = 'individual' # individual, dependent, independent, fine-tune
  test_sub = test_sub
  dl = get_dataloader(file_dir, scheme, test_sub)


  train_dl, test_dl, _= dl
  if i == 1 :
      model = CNN1().to(device)  # CNN1, CNN2(), CNN3, CNN4
  elif i == 2:
      model = CNN2().to(device)
  elif i == 3:
      model = CNN3().to(device)
  elif i == 4:
      model = CNN4().to(device)

  config = {
      'epoch': 500,
      'optimizer': 'Adam',
      'lr': 0.0001,
      'scheme': scheme,
      'subject': test_sub,
      'model': f'CNN{i}',
      'save_path': 'CNN_model/'
  }
  # os.makedirs('/content/gdrive/MyDrive/model', exist_ok=True)
  model.load_state_dict(torch.load(f'CNN_model/sub{test_sub}_CNN{i}_individual_0.0001_8_test16'))
  val_acc = val_model(model, test_dl, device)
  with open("CNN_log.txt", 'a') as f:
      f.writelines(f"sub{config['subject']}_CNN{i}\t{val_acc}\n")
    # loss_record = train_model(model, train_dl, test_dl, device, config)

  # plot confusion matrix
#   model = EEGNet().to(device)
#   model.load_state_dict(torch.load(f"{config['save_path']+config['model']}_{config['scheme']}_{config['lr']}_{BATCH_SIZE}")) # 要改load model的檔名
#   plot_confusion_matrix(model, test_dl, device)
  
def plot_confusion_matrix(model, test_dl, device):
  pred = test_model(model, test_dl, device)
  _, y_test = next(iter(test_dl))
  cm = confusion_matrix(y_test, pred[0].cpu(), normalize = 'pred')
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot()
  plt.plot()


def train_fusion(test_sub):
    device = get_device()
    set_seed(33)
    scheme = 'individual' # individual, dependent, independent, fine-tune
    test_sub = test_sub
    file_dir = 'BCI_data'
    dl = get_dataloader(file_dir, scheme, test_sub)
    train_dl, test_dl, _ = dl
    config = {
            'epoch': 500,
            'optimizer': 'Adam',
            'lr': 0.0001,
            'scheme': scheme,
            'model': 'MCNN',
            'save_path': 'CNN_model/'
        }
    cnn1 = CNN1().to(device)
    cnn1.load_state_dict(torch.load(f"CNN_model/sliding/individual/CNN1_individual_0.0001_128_test16", map_location=torch.device(device)))
    cnn2 = CNN2().to(device)
    cnn2.load_state_dict(torch.load(f"CNN_model/sliding/individual/CNN2_individual_0.0001_128_test16", map_location=torch.device(device)))
    cnn3 = CNN3().to(device)
    cnn3.load_state_dict(torch.load(f"CNN_model/sliding/individual/CNN3_individual_0.0001_128_test16", map_location=torch.device(device)))
    cnn4 = CNN4().to(device)
    cnn4.load_state_dict(torch.load(f"CNN_model/sliding/individual/CNN4_individual_0.0001_128_test16", map_location=torch.device(device)))
    fusion_model = MLP().to(device)
    model = {"CNN1": cnn1,
        "CNN2": cnn2,
        "CNN3": cnn3,
        "CNN4": cnn4,
        "fusion model":fusion_model}

    record = train_fusion_model(model, train_dl, test_dl, device, config)


if __name__ =="__main__":
    for s in range(1,10):
        for i in range(1,5):
            start = time.time()
            main(s,i)
            end = time.time()
            print(f"time cost: {end-start}s")