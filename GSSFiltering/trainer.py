from typing import Union
import torch
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.tester import Tester
import os
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

config = configparser.ConfigParser()
config.read('./config.ini')

if not os.path.exists('./.model_saved'):
    os.mkdir('./.model_saved')

print_num = 25
save_num = int(config['Train']['valid_period'])

lr_split = float(config['Train.Split']['learning_rate'])
lr_kalman = float(config['Train.Kalman']['learning_rate'])

wd_split = float(config['Train.Split']['weight_decay'])
wd_kalman = float(config['Train.Kalman']['weight_decay'])

class Trainer():

    def __init__(self, 
                    dnn:Union[KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2], 
                    data_path, save_path, mode=0):
        # Example:
        #   data_path = './.data/syntheticNL/train/(true)
        #   save_path = '(syntheticNL) Split_KalmanNet.pt'

        self.save_num = save_num

        self.dnn = dnn
        self.x_dim = self.dnn.x_dim
        self.y_dim = self.dnn.y_dim
        self.data_path = data_path
        self.save_path = save_path
        self.mode = mode

        self.loss_best = 1e4

        # self.data_x = torch.load('./.data/syntheticNL/train/(true)state.pt')
        self.data_x = torch.load(data_path + 'state.pt')
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]
        assert(self.x_dim == self.data_x.shape[1])
        assert(self.y_dim == self.data_y.shape[1])
        assert(self.seq_len == self.data_y.shape[2])
        assert(self.data_num == self.data_y.shape[0])

        if self.mode == 0:
            if isinstance(self.dnn, KalmanNet_Filter):
                self.loss_fn = torch.nn.MSELoss()
                # self.loss_fn = torch.nn.SmoothL1Loss()
            elif isinstance(self.dnn, KalmanNet_Filter_v2): 
                self.loss_fn = torch.nn.MSELoss()
                # self.loss_fn = torch.nn.SmoothL1Loss()
        if self.mode == 1:
            self.loss_fn = torch.nn.SmoothL1Loss()
            # self.loss_fn = torch.nn.MSELoss()
        
        if self.mode == 0:
            self.optimizer = torch.optim.Adam(self.dnn.kf_net.parameters(), lr=lr_kalman, weight_decay=wd_kalman)
        if self.mode == 1:
            self.network1 = [self.dnn.kf_net.l1, self.dnn.kf_net.GRU1, self.dnn.kf_net.l2]
            self.network2 = [self.dnn.kf_net.l3, self.dnn.kf_net.GRU2, self.dnn.kf_net.l4]            
            param_group_1 = []
            for elem in self.network1:
                param_group_1 += [{'params': elem.parameters()}]
            param_group_2 = []
            for elem in self.network2:
                param_group_2 += [{'params': elem.parameters()}]
            self.optimizer_list = [torch.optim.Adam(param_group_1, lr=lr_split, weight_decay=wd_split),
                                    torch.optim.Adam(param_group_2, lr=lr_split, weight_decay=wd_split)]
            self.unfreeze_net_current = 1

        cal_num_param = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(cal_num_param(self.dnn.kf_net))

        self.batch_size = int(config['Train']['batch_size'])
        self.alter_num = int(config['Train.Split']['alter_period'])

        self.train_count = 0
        self.data_idx = 0

    def train_batch(self):
        if self.mode == 0:
            self.train_batch_joint()
        elif self.mode == 1:
            if (isinstance(self.dnn, Split_KalmanNet_Filter)):
                self.train_batch_alternative()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def train_batch_alternative(self):

        if self.train_count > 0 and self.train_count % self.alter_num == 0:
            if self.unfreeze_net_current == 1:                        
                self.unfreeze_net_current = 2
            elif self.unfreeze_net_current == 2:           
                self.unfreeze_net_current = 1

        self.optimizer = self.optimizer_list[self.unfreeze_net_current-1]

        self.optimizer.zero_grad()


        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0])
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]            
        batch_x = self.data_x[self.data_idx : self.data_idx+self.batch_size]
        batch_y = self.data_y[self.data_idx : self.data_idx+self.batch_size]

        x_hat = torch.zeros_like(batch_x)            

        for i in range(self.batch_size):
            self.dnn.state_post = batch_x[i,:,0].reshape((-1,1))            
            for ii in range(1, self.seq_len):
                self.dnn.filtering(batch_y[i,:,ii].reshape((-1,1)))
            x_hat[i] = self.dnn.state_history[:,-self.seq_len:]                
            self.dnn.reset(clean_history=False)

        # loss = self.loss_fn(batch_x[:,[0,3],1:], x_hat[:,[0,3],1:])
        loss = self.loss_fn(batch_x[:,:,1:], x_hat[:,:,1:])
        loss.backward()

        ## gradient clipping with maximum value 10
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)

        self.optimizer.step()
        # self.scheduler_list[self.unfreeze_net_current-1].step()   

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            try:
                torch.save(self.dnn.kf_net, './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt')  
            except:
                print('here')
                pass
        if self.train_count % print_num == 1:
            print(f'[Model {self.save_path}] [Train {self.train_count}] loss [dB] = {10*torch.log10(loss):.4f}')


    def train_batch_joint(self):

        self.optimizer.zero_grad()

        if self.data_idx + self.batch_size >= self.data_num:
            self.data_idx = 0
            shuffle_idx = torch.randperm(self.data_x.shape[0])
            self.data_x = self.data_x[shuffle_idx]
            self.data_y = self.data_y[shuffle_idx]            
        batch_x = self.data_x[self.data_idx : self.data_idx+self.batch_size]
        batch_y = self.data_y[self.data_idx : self.data_idx+self.batch_size]

        x_hat = torch.zeros_like(batch_x)            

        for i in range(self.batch_size):
            self.dnn.state_post = batch_x[i,:,0].reshape((-1,1))  
            for ii in range(1, self.seq_len):
                self.dnn.filtering(batch_y[i,:,ii].reshape((-1,1)))
            x_hat[i] = self.dnn.state_history[:,-self.seq_len:]                
            self.dnn.reset(clean_history=False)

        # loss = self.loss_fn(batch_x[:,[0,3],1:], x_hat[:,[0,3],1:])
        loss = self.loss_fn(batch_x[:,:,1:], x_hat[:,:,1:])
        loss.backward()

        ## gradient clipping with maximum value 10
        torch.nn.utils.clip_grad_norm_(self.dnn.kf_net.parameters(), 10)

        self.optimizer.step()

        self.train_count += 1
        self.data_idx += self.batch_size

        if self.train_count % save_num == 0:
            try:
                torch.save(self.dnn.kf_net, './.model_saved/' + self.save_path[:-3] + '_' + str(self.train_count) + '.pt')  
            except:
                print('here')
                pass
        if self.train_count % print_num == 1:
            print(f'[Model {self.save_path}] [Train {self.train_count}] loss [dB] = {10*torch.log10(loss):.4f}')


    def validate(self, tester):            
        if tester.loss.item() < self.loss_best:
            try:
                torch.save(tester.filter.kf_net, './.model_saved/' + self.save_path[:-3] + '_best.pt')
                print(f'Save best model at {self.save_path} & train {self.train_count} & loss [dB] = {tester.loss:.4f}')        
                self.loss_best = tester.loss.item()    
            except:
                pass            
        self.valid_loss = tester.loss.item()
