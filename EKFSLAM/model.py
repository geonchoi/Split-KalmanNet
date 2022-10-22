import torch
import torch.nn as nn

h1 = (5 + 2) * 10 * 8
h2 = (5 * 2) * 1 * 4
nGRU = 2

gru_size_SK = 3
gru_size_K = 5


class DNN_SKalmanNet_SLAM(torch.nn.Module):
    def __init__(self, x_dim:int=5, y_dim:int=2, H1:int=h1, H2:int=h2):

        H1 = 500
        H2 = 40

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim_1 = (self.x_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)
        self.input_dim_2 = (self.y_dim) * 2 + (self.y_dim) + (self.x_dim * self.y_dim)

        self.output_dim_1 = (self.x_dim * self.x_dim) 
        self.output_dim_2 = (self.y_dim * self.y_dim)


        self.gru_input_dim = H1
        self.gru_hidden_dim = round(gru_size_SK * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim)) )   
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1 

        # input layer {x_k - x_{k-1}}
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim_1, H1),
            nn.ReLU()
        )

        # GRU 
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

        l1_out = self.l1(input1)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn1 = self.GRU1(GRU_in, self.hn1)
        l2_out = self.l2(GRU_out)
        Pk = l2_out.reshape((self.x_dim, self.x_dim))

        input2 = torch.cat((observation_inno, diff_obs, linearization_error, Jacobian), axis=0).reshape(-1)
        l3_out = self.l3(input2)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l3_out
        GRU_out, self.hn2 = self.GRU2(GRU_in, self.hn2)
        l4_out = self.l4(GRU_out)
        Sk = l4_out.reshape((self.y_dim, self.y_dim))

        return (Pk, Sk)
 


class DNN_KalmanNet_SLAM(torch.nn.Module):
    def __init__(self, x_dim:int=5, y_dim:int=2, H1:int=h1, H2:int=h2):

        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = (self.x_dim * 2) + (self.y_dim * 2)
        self.output_dim = self.x_dim * self.y_dim

        self.gru_input_dim = H1
        self.gru_hidden_dim = gru_size_K * ((self.x_dim * self.x_dim) + (self.y_dim * self.y_dim))        
        self.gru_n_layer = nGRU
        self.batch_size = 1
        self.seq_len_input = 1 

        # input layer
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, H1),
            nn.ReLU()
        )

        # GRU
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

    def forward(self, state_inno, obs_inno, diff_state, diff_obs):

        input = torch.cat((state_inno, obs_inno, diff_state, diff_obs), axis=0).reshape(-1)
        
        l1_out = self.l1(input)
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.gru_input_dim)
        GRU_in[0,0,:] = l1_out
        GRU_out, self.hn = self.GRU(GRU_in, self.hn)
        l2_out = self.l2(GRU_out)

        kalman_gain = torch.reshape(l2_out, (self.x_dim, self.y_dim))

        return (kalman_gain)   



