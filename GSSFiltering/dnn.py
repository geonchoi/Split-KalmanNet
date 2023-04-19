import torch
import torch.nn as nn
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

nGRU = int(config['DNN.size']['nGRU'])
gru_scale_s = int(config['DNN.size']['gru_scale_s'])
gru_scale_k = int(config['DNN.size']['gru_scale_k'])



class DNN_SKalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2):

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # For NCLT, SyntheticNL (general)
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        # # For Time-Varyling
        # H1 = (x_dim + y_dim) * (5) * 4
        # H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim_1 = (self.x_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)
        self.input_dim_2 = (self.y_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)

        self.output_dim_1 = (self.x_dim * self.x_dim) 
        self.output_dim_2 = (self.y_dim * self.y_dim)

        # input layer {x_k - x_{k-1}}
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, H1),
            nn.ReLU()
        )

        # GRU 
        self.gru_input_dim = H1
        self.gru_hidden_dim = round(gru_scale_s * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)) )      
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1 

        self.hn1 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn1_init = self.hn1.detach().clone()
        self.GRU1 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Pk
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_1)
        )

        # input layer {residual}
        self.l3 = nn.Sequential(
            nn.Linear(self.input_dim_2, H1),
            nn.ReLU()
        )

        # GRU
        self.hn2 = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)     
        self.hn2_init = self.hn2.detach().clone()
        self.GRU2 = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> Sk
        self.l4 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim_2)
        )
        

    def initialize_hidden(self):
        self.hn1 = self.hn1_init.detach().clone()
        self.hn2 = self.hn2_init.detach().clone()

    def forward(self, state_inno, observation_inno, diff_state, diff_obs, linearization_error, Jacobian):

        input1 = torch.cat((state_inno, diff_state, linearization_error, Jacobian), axis=0).reshape(-1)
        input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian), axis=0).reshape(-1)

        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn1 = self.GRU1(GRU_in, self.hn1)
        l2_out = self.l2(GRU_out)
        Pk = l2_out.reshape((self.x_dim, self.x_dim))

        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l3_out
        GRU_out, self.hn2 = self.GRU2(GRU_in, self.hn2)
        l4_out = self.l4(GRU_out)
        Sk = l4_out.reshape((self.y_dim, self.y_dim))

        return (Pk, Sk)


class DNN_KalmanNet_GSS(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2):

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        H1 = (x_dim + y_dim) * (10) * 8
        H2 = (x_dim * y_dim) * 1 * (4)

        self.input_dim = (self.x_dim * 2) + (self.y_dim * 2)
        self.output_dim = self.x_dim * self.y_dim

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, H1),
            nn.ReLU()
        )

        # GRU
        self.gru_input_dim = H1
        self.gru_hidden_dim = gru_scale_k * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim))
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1

        self.hn = torch.randn(self.gru_n_layer, self.batch_size, self.gru_hidden_dim)
        self.hn_init = self.hn.detach().clone()
        self.GRU = nn.GRU(self.gru_input_dim, self.gru_hidden_dim, self.gru_n_layer)

        # GRU output -> H2 -> kalman gain
        self.l2 = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, H2),
            nn.ReLU(),
            nn.Linear(H2, self.output_dim)
        )
        

    def initialize_hidden(self):
        self.hn = self.hn_init.detach().clone()

    def forward(self, state_inno, observation_inno, diff_state, diff_obs):

        input = torch.cat((state_inno, observation_inno, diff_state, diff_obs), axis=0).reshape(-1)
        l1_out = self.l1(input)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn = self.GRU(GRU_in, self.hn)
        l2_out = self.l2(GRU_out)

        kalman_gain = torch.reshape(l2_out, (self.x_dim, self.y_dim))

        return (kalman_gain)      


class KNet_architecture_v2(torch.nn.Module):
    def __init__(self, x_dim:int=2, y_dim:int=2, in_mult=20, out_mult=40):
        super().__init__()

        self.gru_num_param_scale = 1

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.gru_n_layer = 1
        self.batch_size = 1
        self.seq_len_input = 1 # Forward 전후로 처리해야할게 있으니 hidden 초기화를 직접 하고 시퀀스 길이를 1로!

        self.prior_Q = torch.eye(x_dim)
        self.prior_Sigma = torch.randn((x_dim,x_dim))
        self.prior_S = torch.randn((y_dim,y_dim))

        # GRU to track Q (5 x 5)
        self.d_input_Q = self.x_dim * in_mult
        self.d_hidden_Q = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q)
        self.h_Q = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Q)
        # self.h_Q_init = self.h_Q.detach().clone()

        # GRU to track Sigma (5 x 5)
        self.d_input_Sigma = self.d_hidden_Q + self.x_dim * in_mult
        self.d_hidden_Sigma = self.gru_num_param_scale * self.x_dim ** 2
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma)
        self.h_Sigma = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_Sigma)
        # self.h_Sigma_init = self.h_Sigma.detach().clone()

        # GRU to track S (2 x 2)
        self.d_input_S = self.y_dim ** 2 + 2 * self.y_dim * in_mult
        self.d_hidden_S = self.gru_num_param_scale * self.y_dim ** 2
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S)
        self.h_S = torch.randn(self.gru_n_layer, self.batch_size, self.d_hidden_S)
        # self.h_S_init = self.h_S.detach().clone()

        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.y_dim ** 2
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU())

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.y_dim * self.x_dim
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2))

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.x_dim ** 2
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU())

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU())
        
        # Fully connected 5
        self.d_input_FC5 = self.x_dim
        self.d_output_FC5 = self.x_dim * in_mult
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU())

        # Fully connected 6
        self.d_input_FC6 = self.x_dim
        self.d_output_FC6 = self.x_dim * in_mult
        self.FC6 = nn.Sequential(
                nn.Linear(self.d_input_FC6, self.d_output_FC6),
                nn.ReLU())
        
        # Fully connected 7
        self.d_input_FC7 = 2 * self.y_dim
        self.d_output_FC7 = 2 * self.y_dim * in_mult
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU())

    def initialize_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(1, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S[0, 0, :] = self.prior_S.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma[0, 0, :] = self.prior_Sigma.flatten()
        hidden = weight.new(1, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q[0, 0, :] = self.prior_Q.flatten()

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1])
            expanded[0, 0, :] = x
            return expanded

        obs_diff = expand_dim(obs_diff.reshape((-1)))
        obs_innov_diff = expand_dim(obs_innov_diff.reshape((-1)))
        fw_evol_diff = expand_dim(fw_evol_diff.reshape((-1)))
        fw_update_diff = expand_dim(fw_update_diff.reshape((-1)))

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = fw_evol_diff
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)

        """
        # FC 8
        in_FC8 = out_Q
        out_FC8 = self.FC8(in_FC8)
        """

        # FC 6
        in_FC6 = fw_update_diff
        out_FC6 = self.FC6(in_FC6)

        # Sigma_GRU
        in_Sigma = torch.cat((out_Q, out_FC6), 2)
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = torch.cat((obs_diff, obs_innov_diff), 2)
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2.reshape((self.x_dim, self.y_dim))
