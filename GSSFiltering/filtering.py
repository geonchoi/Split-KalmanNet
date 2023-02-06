from GSSFiltering.dnn import DNN_KalmanNet_GSS, DNN_SKalmanNet_GSS, KNet_architecture_v2
from GSSFiltering.model import GSSModel
import torch
import torch.nn.functional as F

class Extended_Kalman_Filter():
    
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state
        self.Q = GSSModel.cov_q
        self.R = GSSModel.cov_r
        
        self.init_cov = torch.zeros((self.x_dim, self.x_dim))
        self.state_history = self.init_state.detach().clone()
        self.reset(clean_history=True)   

    def reset(self, clean_history=False):
        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        if clean_history:
            self.state_history = self.init_state.detach().clone()     
            self.cov_trace_history = torch.zeros((1,))

    def filtering(self, observation):
        with torch.no_grad():
            # print(self.GSSModel.r2)
            # observation: column vector
            x_last = self.state_post
            x_predict = self.GSSModel.f(x_last)

            y_predict = self.GSSModel.g(x_predict)
            residual = observation - y_predict

            F_jacob = self.GSSModel.Jacobian_f(x_last)
            H_jacob = self.GSSModel.Jacobian_g(x_predict)
            cov_pred = (F_jacob @ self.cov_post @ torch.transpose(F_jacob, 0, 1)) + self.Q

            K_gain = cov_pred @ torch.transpose(H_jacob, 0, 1) @ \
                torch.linalg.inv(H_jacob@cov_pred@torch.transpose(H_jacob, 0, 1) + self.R)

            x_post = x_predict + (K_gain @ residual)

            cov_post = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ cov_pred
            cov_trace = torch.trace(cov_post)

            self.state_post = x_post.detach().clone()
            self.cov_post = cov_post.detach().clone()
            self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)    
            self.cov_trace_history = torch.cat((self.cov_trace_history, cov_trace.reshape(-1).clone()))

            self.pk = cov_pred

class KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_KalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)
        

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()  
            self.cov_trace_history = torch.zeros((1,))      
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
          

    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(state_inno, residual, diff_state, diff_obs)

        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # K_gain = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in)


        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)


class KalmanNet_Filter_v2():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = KNet_architecture_v2(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)
        

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.cov_trace_history = torch.zeros((1,))     
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)


    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(diff_obs, residual, diff_state, state_inno)
        
        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # K_gain = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in)  

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)



class Split_KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_SKalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        # self.state_history = self.init_state.detach().clone()
        self.reset(clean_history=True)
  

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()        
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)


    def filtering(self, observation):
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)
        residual = observation - y_predict

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past
        ## input 6: Jacobian
        H_jacob = self.GSSModel.Jacobian_g(x_predict)     
        ## input 5: linearization error
        # linearization_error = H_jacob@x_predict
        linearization_error = y_predict - H_jacob@x_predict


        H_jacob_in = H_jacob.reshape((-1,1))
        # H_jacob_in = F.normalize(H_jacob_in, p=2, dim=0, eps=1e-12)
        linearization_error_in = linearization_error
        # linearization_error_in = F.normalize(linearization_error, p=2, dim=0, eps=1e-12)
        (Pk, Sk) = self.kf_net(state_inno, residual, diff_state, diff_obs, linearization_error_in, H_jacob_in)

        # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
        # residual_in = F.normalize(residual, p=2, dim=0, eps=1e-12)
        # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
        # diff_obs_in = F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
        # # residual_in = residual
        # # diff_obs_in = diff_obs
        # (Pk, Sk) = self.kf_net(state_inno_in, residual_in, diff_state_in, diff_obs_in, linearization_error, H_jacob.reshape((-1,1)))

        K_gain = Pk @ torch.transpose(H_jacob, 0, 1) @ Sk

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)