import torch
import os
import numpy as np
import math, random

if not os.path.exists('./.data'):
    os.mkdir('./.data')


class GSSModel():
    def __init__(self):
        self.cov_q = None
        self.cov_r = None
        self.x_dim = None
        self.y_dim = None

    def generate_data(self):
        raise NotImplementedError()

    def f(self, current_state):
        raise NotImplementedError()
    def g(self, current_state):
        raise NotImplementedError()
    def Jacobian_f(self, x):
        raise NotImplementedError()
    def Jacobian_g(self, x):
        raise NotImplementedError()

    def next_state(self, current_state):
        noise = np.random.multivariate_normal(np.zeros(self.x_dim), self.cov_q)
        return self.f(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))
    def observe(self, current_state):
        noise = np.random.multivariate_normal(np.zeros(self.y_dim), self.cov_r)
        return self.g(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))


class SyntheticNLModel(GSSModel):
    def __init__(self, is_train, is_mismatch=False):
        super().__init__()
        self.is_train = is_train
        self.is_mismatch = is_mismatch

        self.x_dim = 2
        self.y_dim = 2

        self.q2_dB = -30
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.v_dB = 20
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = self.r2 * torch.eye(self.y_dim)

        self.theta = 10 * 2 * math.pi/360
        self.F = torch.tensor(
                    [[math.cos(self.theta), -math.sin(self.theta)],
                     [math.sin(self.theta), math.cos(self.theta)]])    

        self.init_state = torch.tensor([1., 0.]).reshape((-1,1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))


    def generate_data(self):
        self.save_path = './.data/syntheticNL/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.is_train:
            self.seq_len = 20
            self.num_data = 500
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        else:
            self.seq_len = 50
            self.num_data = 100
            self.save_path += 'test/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        if self.is_mismatch:
            self.save_path += '(mismatch)'
        else:
            self.save_path += '(true)'

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))
        with torch.no_grad():
            for i in range(self.num_data):
                if self.is_train:
                    theta = torch.distributions.uniform.Uniform(0,2*torch.pi).sample()
                    dr = torch.tensor(1.)
                    self.init_state = torch.sqrt(dr)*torch.tensor([torch.cos(theta), torch.sin(theta)]).reshape((-1,1))
                if i % 100 == 0 :
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len))
                obs_tmp = torch.zeros((self.y_dim, self.seq_len))
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_tmp[:,j] = x.reshape(-1)
                    obs_tmp[:,j] = y.reshape(-1)
                
                state_mtx[i] = state_tmp
                obs_mtx[i] = obs_tmp
        
        torch.save(state_mtx, self.save_path + 'state.pt')
        torch.save(obs_mtx, self.save_path + 'obs.pt')                 

    def f(self, x):
        # x는 column vector
        return torch.matmul(self.F, x)                          
    
    def g(self, x):
        return x.reshape((-1,1))
        # x = x.reshape(-1)
        # y1 = torch.sqrt(x[0]**2 + x[1]**2)
        # y2 = torch.arctan2(x[0], x[1]) 
        # return torch.tensor([y1, y2]).reshape((-1,1))

    def Jacobian_f(self, x):
        return self.F
        
    def Jacobian_g(self, x):
        return torch.tensor([[1., 0.], [0., 1.]])
        # H11 = x[0]/torch.sqrt(x[0]**2 + x[1]**2)
        # H12 = x[1]/torch.sqrt(x[0]**2 + x[1]**2)
        # H21 = x[1]/torch.sqrt(x[0]**2 + x[1]**2)
        # H22 = -x[0]/torch.sqrt(x[0]**2 + x[1]**2)
        # return torch.tensor([[H11, H12], [H21, H22]])  
