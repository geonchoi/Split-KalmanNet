import math
import torch
from GSSFiltering.filtering import Extended_Kalman_Filter, KalmanNet_Filter, KalmanNet_Filter_v2, Split_KalmanNet_Filter
import time
from datetime import timedelta

print_num = 25

class Tester():
    def __init__(self, filter, data_path, model_path, is_validation=False, is_mismatch=False):
        # Example:
        #   data_path = './.data/syntheticNL/test/(true)
        #   model_path = './.model_saved/(syntheticNL) Split_KalmanNet_5000.pt'

        if isinstance(filter, Extended_Kalman_Filter):
            self.result_path = 'EKF '
        if isinstance(filter, KalmanNet_Filter):
            self.result_path = 'KF v1 '
        if isinstance(filter, KalmanNet_Filter_v2):
            self.result_path = 'KF v2 '
        if isinstance(filter, Split_KalmanNet_Filter):
            self.result_path = 'SKF '


        self.filter = filter
        if not isinstance(filter, Extended_Kalman_Filter):
            self.filter.kf_net = torch.load(model_path)
        self.x_dim = self.filter.x_dim
        self.y_dim = self.filter.y_dim
        self.data_path = data_path
        self.model_path = model_path
        self.is_validation = is_validation
        self.is_mismatch = is_mismatch

        self.loss_fn = torch.nn.MSELoss()

        self.data_x = torch.load(data_path + 'state.pt')
        self.data_y = torch.load(data_path + 'obs.pt')
        self.data_num = self.data_x.shape[0]
        self.seq_len = self.data_x.shape[2]
        assert(self.x_dim == self.data_x.shape[1])
        assert(self.y_dim == self.data_y.shape[1])
        assert(self.seq_len == self.data_y.shape[2])
        assert(self.data_num == self.data_y.shape[0])

        x_hat = torch.zeros_like(self.data_x)

        start_time = time.monotonic()

        with torch.no_grad():
            for i in range(self.data_num):
                if i % print_num == 0:
                    if self.is_validation:
                        print(f'Validating {i} / {self.data_num} of {self.model_path}')
                    else:
                        print(f'Testing {i} / {self.data_num} of {self.model_path}')
                
                self.filter.state_post = self.data_x[i,:,0].reshape((-1,1))  ### !!!! 추가한거!!
                if not self.is_mismatch:
                    self.filter.GSSModel.set_v_dB(30)
                    self.filter.R = self.filter.GSSModel.cov_r
                for ii in range(1, self.seq_len):
                    if not self.is_mismatch:
                        # if ii % 100 == 1:
                        if ii % 10 == 1:
                            # print(f'Before v_dB = {self.filter.GSSModel.v_dB}')
                            # self.filter.GSSModel.set_v_dB((self.filter.GSSModel.v_dB + 10) % 50)
                            self.filter.GSSModel.set_v_dB((self.filter.GSSModel.v_dB + 1) % 50)
                            self.filter.R = self.filter.GSSModel.cov_r
                            # print(f'After v_dB = {self.filter.GSSModel.v_dB}')

                    self.filter.filtering(self.data_y[i,:,ii].reshape((-1,1)))
                x_hat[i] = self.filter.state_history[:,-self.seq_len:]                
                self.filter.reset(clean_history=False)

            end_time = time.monotonic()
            print(timedelta(seconds=end_time - start_time))

            torch.save(x_hat, data_path + self.result_path + 'x_hat.pt')

            loss = self.loss_fn(self.data_x[:,:,1:], x_hat[:,:,1:])
            # loss = self.loss_fn(self.data_x[:,[0,3],1:], x_hat[:,[0,3],1:])
            loss_dB = 10*torch.log10(loss)
            print(f'loss [dB] = {loss_dB:.4f}')


            # Compute loss at instantaneous time
            self.loss_instant = torch.zeros(self.data_x[:,:,1:].shape[-1])
            for i in range(self.data_x[:,:,1:].shape[-1]):
                self.loss_instant[i] = self.loss_fn(self.data_x[:, :, i+1], x_hat[:, :, i+1])
            self.loss_instant_dB = 10*torch.log10(self.loss_instant)

        self.loss = loss_dB