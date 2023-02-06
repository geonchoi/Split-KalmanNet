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


class NCLT(GSSModel):
    def __init__(self, mode='train'):
        super().__init__()
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')

        self.x_dim = 6
        self.y_dim = 2

        dt = 1.
        self.F = torch.tensor([[1., dt, 1/2*dt**2],
                               [0., 1., dt],
                               [0., 0., 1.]])
        self.H = torch.tensor([[0., 1., 0.]])

        self.q2_dB = 0
        self.r2_dB = 0
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.r2 = torch.tensor(10**(self.r2_dB/10))
        self.cov_q = self.q2 * torch.tensor([[1/4*dt**4, 1/2*dt**3, 1/2*dt**2],
                                             [1/2*dt**3, dt**2,     dt],
                                             [1/2*dt**2, dt,        1.]])
        self.cov_r = self.r2 * torch.eye(self.y_dim)

        self.F     = torch.kron(torch.eye(2), self.F)
        self.H     = torch.kron(torch.eye(2), self.H)
        self.cov_q = torch.kron(torch.eye(2), self.cov_q)

        self.init_state = torch.tensor([0., 0., 0., 0, 0., 0.]).reshape((-1,1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))

    def generate_data(self):
        print('not necessary')        

    def f(self, x):
        # x는 column vector
        return torch.matmul(self.F, x)                          
    
    def g(self, x):
        return torch.matmul(self.H, x)                            

    def Jacobian_f(self, x):
        return self.F
        
    def Jacobian_g(self, x):
        return self.H     


class SyntheticNL_mismatched_Model(GSSModel):
    def __init__(self, mode='train'):
        super().__init__()
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')

        self.x_dim = 2
        self.y_dim = 2

        self.q2_dB = -30
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.v_dB = 30
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = self.r2 * torch.eye(self.y_dim)

        self.theta = 10 * 2 * math.pi/360
        self.F = torch.tensor(
                    [[math.cos(self.theta), -math.sin(self.theta)],
                     [math.sin(self.theta), math.cos(self.theta)]])
        
        self.dtheta = 0 * 2 * math.pi / 360
        self.H = torch.tensor(
                    [[math.cos(self.dtheta), -math.sin(self.dtheta)],
                     [math.sin(self.dtheta), math.cos(self.dtheta)]])

        self.init_state = torch.tensor([1., 0.]).reshape((-1,1))
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))

    def set_q2_dB(self, q2_dB):
        self.q2_dB = q2_dB
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.cov_q = self.q2 * torch.eye(self.x_dim)
    
    def set_v_dB(self, v_dB):
        self.v_dB = v_dB
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_r = self.r2 * torch.eye(self.y_dim)

    def generate_data(self):
        self.save_path = './.data/syntheticMMNL/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.mode == 'train':
            self.seq_len = 15
            self.num_data = 500
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        elif self.mode == 'valid':
            self.seq_len = 50
            self.num_data = 50
            self.save_path += 'valid/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        elif self.mode == 'test':
            self.seq_len = 1000
            self.num_data = 1
            self.save_path += 'test/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        else:
            raise NotImplementedError()

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))
        with torch.no_grad():
            for i in range(self.num_data):
                theta = torch.distributions.uniform.Uniform(0,2*torch.pi).sample()
                dr = torch.tensor(1.)
                self.init_state = torch.sqrt(dr)*torch.tensor([torch.cos(theta), torch.sin(theta)]).reshape((-1,1))

                # if self.mode == 'train':
                #     self.update_dtheta(torch.distributions.uniform.Uniform(-20, 20).sample())

                if self.mode in ['train', 'valid']:
                    self.set_v_dB(random.randint(0,3)*10)
                else:
                    self.set_v_dB(30)

                if i % 100 == 0 :
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len))
                obs_tmp = torch.zeros((self.y_dim, self.seq_len))
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    if self.mode == 'test':
                        # if j % 100 == 1:
                        if j % 10 == 1:
                            # print(f'Before v_dB = {self.v_dB}')
                            # self.set_v_dB((self.v_dB + 10) % 50)
                            self.set_v_dB((self.v_dB + 1) % 50)
                            # print(f'After v_dB = {self.v_dB}')

                    if self.mode == 'train':
                        if j % 2 == 1:
                            self.set_v_dB((self.v_dB + 1) % 50)
                            
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
        x = x.reshape(-1)
        x = torch.matmul(self.H, x)
        
        # return x.reshape((-1,1))
        
        y1 = torch.sqrt(x[0]**2 + x[1]**2)
        y2 = torch.arctan2(x[1], x[0]) 
        return torch.tensor([y1, y2]).reshape((-1,1))

    def Jacobian_f(self, x):
        return self.F
        
    def Jacobian_g(self, x):
        # return torch.tensor([[1., 0.], [0., 1.]])

        H11 = x[0]/torch.sqrt(x[0]**2 + x[1]**2)
        H12 = x[1]/torch.sqrt(x[0]**2 + x[1]**2)
        H21 = -x[1]/(x[0]**2 + x[1]**2)
        H22 = x[0]/(x[0]**2 + x[1]**2)
        return torch.tensor([[H11, H12], [H21, H22]])  

    def update_dtheta(self, theta_deg):
        self.dtheta = theta_deg * 2 * math.pi / 360
        self.H = torch.tensor(
                    [[math.cos(self.dtheta), -math.sin(self.dtheta)],
                     [math.sin(self.dtheta), math.cos(self.dtheta)]])


class SyntheticNLModel(GSSModel):
    def __init__(self, mode='train'):
        super().__init__()
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')

        self.x_dim = 2
        self.y_dim = 2

        self.q2_dB = -30
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.v_dB = 40
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

        if self.mode == 'train':
            self.seq_len = 15
            self.num_data = 500
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        elif self.mode == 'valid':
            self.seq_len = 50
            self.num_data = 50
            self.save_path += 'valid/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        elif self.mode == 'test':
            self.seq_len = 100
            self.num_data = 10
            self.save_path += 'test/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        else:
            raise NotImplementedError()

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))
        with torch.no_grad():
            for i in range(self.num_data):
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
        # y2 = torch.arctan2(x[1], x[0]) 
        # return torch.tensor([y1, y2]).reshape((-1,1))

    def Jacobian_f(self, x):
        return self.F
        
    def Jacobian_g(self, x):
        return torch.tensor([[1., 0.], [0., 1.]])
        # H11 = x[0]/torch.sqrt(x[0]**2 + x[1]**2)
        # H12 = x[1]/torch.sqrt(x[0]**2 + x[1]**2)
        # H21 = -x[1]/(x[0]**2 + x[1]**2)
        # H22 = x[0]/(x[0]**2 + x[1]**2)
        # return torch.tensor([[H11, H12], [H21, H22]])  


import math, random
import numpy as np
class LorenzAttractor(GSSModel):
    def __init__(self, mode='train'):
        super().__init__()
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')

        self.is_nonlinear = True

        self.x_dim = 3
        self.y_dim = 3

        self.q2_dB = -30
        self.q2 = torch.tensor(10**(self.q2_dB/10))
        self.v_dB = 50
        self.v = 10**(self.v_dB/10)
        self.r2 = torch.mul(self.q2, self.v)
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = self.r2 * torch.eye(self.y_dim)

        self.J = 5
        self.dt = 0.01

        self.init_state = torch.tensor([1, 1, 1]).reshape((-1,1)).float()
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))


    def generate_data(self):
        self.save_path = './.data/lorenz/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.mode == 'train':
            # 2000 / 100 * 100 = 2000 
            self.num_data_total = 1000
            self.seq_len = 2000
            if self.is_nonlinear:
                # self.chunk_len = 50
                self.seq_len = 500
                self.chunk_len = 100
            else:
                self.chunk_len = 100
            self.chunk_num = int(self.seq_len / self.chunk_len)
            self.num_data = int(self.num_data_total / self.chunk_num)
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        elif self.mode == 'valid':
            if self.is_nonlinear:
                self.seq_len = 100
            else:
                self.seq_len = 200
            self.num_data = 2
            self.save_path += 'valid/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        elif self.mode == 'test':
            if self.is_nonlinear:
                # self.seq_len = 500
                self.seq_len = 1000
            else:
                self.seq_len = 2000
            self.num_data = 1
            self.save_path += 'test/'      
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        else:
            raise NotImplementedError()


        state_mtx_before_chunk = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx_before_chunk = torch.zeros((self.num_data, self.y_dim, self.seq_len))
        with torch.no_grad():
            for i in range(self.num_data):
                if i % 5 == 0 :
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
                
                state_mtx_before_chunk[i] = state_tmp
                obs_mtx_before_chunk[i] = obs_tmp
        
        if self.mode == 'train':
            # chunking
            state_mtx = torch.zeros((self.num_data_total, self.x_dim, self.chunk_len))
            obs_mtx = torch.zeros((self.num_data_total, self.y_dim, self.chunk_len))

            j_out = 0
            for i in range(self.num_data):
                state_chunk = torch.split(state_mtx_before_chunk[i], self.chunk_len, dim=1)
                obs_chunk = torch.split(obs_mtx_before_chunk[i], self.chunk_len, dim=1)

                permutation_idx = np.random.permutation(self.chunk_num)
                for j in range(self.chunk_num):
                    state_mtx[j_out + j] = state_chunk[permutation_idx[j]]
                    obs_mtx[j_out + j] = obs_chunk[permutation_idx[j]] 

                    # state_mtx[j_out + j] = state_chunk[j]
                    # obs_mtx[j_out + j] = obs_chunk[j]
                                
                j_out += self.chunk_num
        else:
            state_mtx = state_mtx_before_chunk
            obs_mtx = obs_mtx_before_chunk

        torch.save(state_mtx, self.save_path + 'state.pt')
        torch.save(obs_mtx, self.save_path + 'obs.pt')                 

    def f(self, x):
        # x는 column vector
        A = torch.tensor([
            [-10, 10, 0],
            [28-x[2], -1, 0],
            [x[1], 0, -8/3]
        ]).float()

        F = torch.eye(3)
        for j in range(1, self.J+1):
            F_add = (torch.matrix_power(A*self.dt, j) / math.factorial(j))
            F = torch.add(F, F_add)
            
        return torch.matmul(F, x)
    
    def g(self, x):  
        if self.is_nonlinear:
            return self.cart_to_spherical(x)
        else:
            return x
        

    def Jacobian_f(self, x):
        # x는 column vector
        A = torch.tensor([
            [-10, 10, 0],
            [28-x[2], -1, 0],
            [x[1], 0, -8/3]
        ]).float()

        F = torch.eye(3)
        for j in range(1, self.J+1):
            F_add = (torch.matrix_power(A*self.dt, j) / math.factorial(j))
            F = torch.add(F, F_add)
            
        return F     

    def Jacobian_g(self, x):

        if self.is_nonlinear:
            rho = torch.norm(x, p=2).view(1,1)
            x2py2 = (x[0]**2 + x[1]**2)
            return torch.tensor([
                [x[0]/rho, x[1]/rho, x[2]/rho],
                [x[0]*x[2] / rho/rho/torch.sqrt(x2py2), x[1]*x[2]/rho/rho/torch.sqrt(x2py2), -x2py2/rho/rho/torch.sqrt(x2py2)],
                [-x[1]/x2py2, x[0]/x2py2, 0]
            ]).float()
        else:
            return torch.eye(3)


    def cart_to_spherical(self, cart):
        rho = torch.norm(cart, p=2).view(1,1)
        phi = torch.atan2(cart[1, ...], cart[0, ...]).view(1,1)
        phi = phi + (phi < 0).type_as(phi) * (2 * torch.pi)

        theta = torch.acos(cart[2, ...] / rho).view(1,1)

        spher = torch.cat([rho, theta, phi], dim=0)

        return spher