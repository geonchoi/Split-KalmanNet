import random
import numpy as np
import torch
from .model import DNN_KalmanNet_SLAM
from .model import DNN_SKalmanNet_SLAM
import copy
import torch.nn.functional as F

class Robot():
    def __init__(self, is_dnn=False, is_test=False, mode=0, is_mismatch=False):
        
        self.is_manual = False

        self.mode = mode
        self.is_dnn = is_dnn
        self.is_test = is_test
        self.is_mismatch = is_mismatch
        self.dnn_first = False 
        self.first_filtering = True

        self.fill = 1e3
        self.eps = 0

        self.X_init = [0]
        self.Y_init = [0]
        self.Angle_init = [0]
        self.velocity = 5

        if self.is_manual:
            self.X_init = [-20]
            self.Y_init = [-20]

        # train
        self.sig_proc_range = [np.sqrt(5e-4).item(), np.sqrt(5e-3).item()]
        self.sig_obs_range = [np.sqrt(5e-4).item(), np.sqrt(5e-3).item()]                                                                      
        if is_test:          
            if not is_mismatch:
                self.sig_proc_range = [np.sqrt(1e-3).item(), np.sqrt(1e-3).item()]            
                self.sig_obs_range = [np.sqrt(1e-3).item(), np.sqrt(1e-3).item()]
            else:
                self.sig_proc_range = [np.sqrt(1e-4).item(), np.sqrt(1e-4).item()]           
                self.sig_obs_range = [np.sqrt(1e-4).item(), np.sqrt(1e-4).item()]    
                     

        self.sig_proc = np.random.uniform(self.sig_proc_range[0], self.sig_proc_range[1])
        self.sig_obs = np.random.uniform(self.sig_obs_range[0], self.sig_obs_range[1])

        # assumed value for EKF (exact)
        self.sig_proc_known = self.sig_proc
        self.sig_obs_known = self.sig_obs   

        # assumed value for EKF (mismatch)
        self.sig_proc_mismatch = np.sqrt(1e-3).item()
        self.sig_obs_mismatch = np.sqrt(1e-3).item()
   

        self.X = self.X_init.copy()         # X trajectory (true)
        self.Y = self.Y_init.copy()         # Y trajectory (true)
        self.Angle = self.Angle_init.copy() # radian... (true)

        # train
        self.num_landmark = 2
        self.region_max = 10
        # self.num_landmark = 5
        # self.region_max = 30
        if is_test:       
    
            # self.num_landmark = 15
            # self.region_max = 20
            # self.num_landmark = 5
            self.region_max = 30    

            self.num_landmark = 10      

            # test velocity
            self.velocity = 1

        if self.is_manual:
            self.control_input = ['w'] * 7 + ['a']
            self.control_direction = {'w':0, 'a':90, 's':180, 'd':270}
            self.move_count = 0
            self.velocity = 3

        self.velocity_true = self.velocity

        if is_mismatch:
            self.rotate_offset = (torch.pi / 180) * 10
        
        if is_test:
            if not is_mismatch:
                self.a = 1 * 1e3
                self.b = 1e1
            else:
                self.a = 1e0
                self.b = 1e0
            # self.a1 = 1e4
            # self.a2 = 1e6
            self.a1 = self.a
            self.a2 = self.a
        else:
            self.a = 1e3
            self.b = 1e1
        a_known = 1e2
        b_known = 1e1
    

        self.generate_landmark()
        self.landmarks_visited = np.zeros(self.num_landmark)

        if not is_dnn:
            size_tmp = 3+2*self.num_landmark
            Q = np.zeros((size_tmp, size_tmp))
            if mode == 0:
                Q[0:3, 0:3] = (self.sig_proc_known**2) * np.array([[self.b, 0, 0], [0, self.b, 0], [0, 0, 1]])
                R = (self.sig_obs_known**2) * np.array([[self.a, 0], [0, 1]])
            else:
                Q[0:3, 0:3] = (self.sig_proc_mismatch**2) * np.array([[b_known, 0, 0], [0, b_known, 0], [0, 0, 1]])
                R = (self.sig_obs_mismatch**2) * np.array([[a_known, 0], [0, 1]])                
            self.EKF_initialization(Q, R)                                                             
        else:
            self.dnn_first = True            
            self.KF_net_initialization()

        self.state_history_init = copy.deepcopy(self.state_history) 

    def KF_net_initialization(self):
        self.dnn_first = True 
        size = 3 + 2*self.num_landmark
        self.state_post = torch.zeros(size).reshape((-1,1))
        self.state_post[0][0] = self.X[-1]
        self.state_post[1][0] = self.Y[-1]
        self.state_post[2][0] = self.Angle[-1]    

        if self.mode in [0.1]:
            self.kf_net = DNN_KalmanNet_SLAM()
        elif self.mode == 1.3:
            self.kf_net = DNN_SKalmanNet_SLAM() 
        else:
            raise NotImplementedError()
        self.kf_net.initialize_hidden()
        self.state_history = torch.Tensor(self.state_post) 

    def EKF_initialization (self, cov_processing, cov_observation):
        size = 3 + 2*self.num_landmark
        self.state_post = np.zeros(size).reshape((-1,1))
        self.state_post[0][0] = self.X[-1] # deep copy ok
        self.state_post[1][0] = self.Y[-1]
        self.state_post[2][0] = self.Angle[-1]
        self.cov_post = np.zeros((size, size))
        np.fill_diagonal(self.cov_post[3:,3:], self.fill)

        self.cov_processing = cov_processing
        self.cov_observation = cov_observation
        self.state_history = self.state_post                           


    def reset(self, clean_history=False):
        self.sig_proc = np.random.uniform(self.sig_proc_range[0], self.sig_proc_range[1])
        self.sig_obs = np.random.uniform(self.sig_obs_range[0], self.sig_obs_range[1])
        size = 3 + 2*self.num_landmark
        self.X = copy.deepcopy(self.X_init)
        self.Y = copy.deepcopy(self.Y_init)
        self.Angle = copy.deepcopy(self.Angle_init)
        if self.is_dnn:
            self.dnn_first = True
            self.kf_net.initialize_hidden()
            self.state_post = torch.zeros(size).reshape((-1,1))                           
        else:
            self.state_post = np.zeros(size).reshape((-1,1))
            self.cov_post = np.zeros((size, size))
            np.fill_diagonal(self.cov_post[3:,3:], self.fill)     
                

        self.state_post[0][0] = self.X[-1]
        self.state_post[1][0] = self.Y[-1]
        self.state_post[2][0] = self.Angle[-1]  

        if self.is_dnn:
            self.state_history = torch.cat((self.state_history, self.state_post), axis=1) # 얘가 학습할때 쓸거
        else:
            self.state_history = np.concatenate((self.state_history, self.state_post), axis=1)

        self.landmarks_visited = np.zeros(self.num_landmark)
        self.generate_landmark()

        if clean_history:
            self.clean_history()

        self.first_filtering = True

    def clean_history(self):
        self.state_history = copy.deepcopy(self.state_history_init)

    def estimate_state(self, observation):
        if not self.is_dnn:
            self.EKF_filtering(observation)
        else:
            if self.mode in [0.1]:
                self.KF_net_filtering(observation)
            elif self.mode in [1.3]:
                self.KF_net_filtering_split(observation)
            else: 
                raise NotImplementedError()
        self.first_filtering = False



###############################################################################################################
###############################################################################################################
###############################################################################################################
    def move(self):
        # self.sig_proc = np.random.uniform(self.sig_proc_range[0], self.sig_proc_range[1])
        # self.sig_obs = np.random.uniform(self.sig_obs_range[0], self.sig_obs_range[1])      

        self.dtheta = np.random.rand(1).item() * 2*np.pi
        if self.is_manual:
            self.dtheta = self.control_direction[self.control_input[self.move_count % len(self.control_input)]] * torch.pi / 180
            self.move_count += 1

        x_current = self.X[-1]
        y_current = self.Y[-1]
        angle_current = self.Angle[-1]

        if self.is_mismatch:
            # velocity_move = self.velocity_true + np.sqrt(1e-1)*np.random.randn(1).item()
            # velocity_move = self.velocity_true + 0.5
            velocity_move = self.velocity_true
        else:
            velocity_move = self.velocity_true             

        x = x_current + np.cos(angle_current) * (velocity_move)
        y = y_current + np.sin(angle_current) * (velocity_move) 

        x += np.random.randn(1).item() * self.sig_proc * np.sqrt(self.b)
        y += np.random.randn(1).item() * self.sig_proc * np.sqrt(self.b)

        self.X.append(x)
        self.Y.append(y)

        angle = (angle_current + self.dtheta + 
            np.random.randn(1).item() * self.sig_proc) % (2*np.pi)

        if angle > np.pi:
            angle = angle - 2*np.pi
        self.Angle.append(angle)
        return x, y, angle


    def observe(self): 
        ''' observation : np.array'''
        def cal_observation(landmark):
            x_current = self.X[-1]
            y_current = self.Y[-1]
            angle_current = self.Angle[-1]

            x_rm = landmark[0]
            y_rm = landmark[1]

            dx = x_rm-x_current
            dy = y_rm-y_current

            range = np.sqrt(dx**2 + dy**2)
            bearing = np.arctan2(dy, dx) 
            
            # if self.is_mismatch:
                # bearing += self.rotate_offset

            if self.is_test:
                a = np.random.uniform(self.a1, self.a2)                 
                range += np.random.randn(1).item() * self.sig_obs * np.sqrt(a)
            else:
                range += np.random.randn(1).item() * self.sig_obs * np.sqrt(self.a)            
            bearing += np.random.randn(1).item() * self.sig_obs
            bearing -= angle_current

            bearing %= 2*np.pi
            if bearing > np.pi:
                bearing -= 2*np.pi
            
            return np.array([range, bearing])

        self.observation = np.zeros((self.num_landmark, 2))

        # # swapped data association simulation?
        # if self.is_mismatch:
        #     if np.random.rand() < 0.01:
        #         idx_tmp = np.random.choice(len(self.landmarks), 2, replace=False)
        #         idx_set = np.arange(len(self.landmarks))
        #         idx_set[idx_tmp[0]] = idx_tmp[1]
        #         idx_set[idx_tmp[1]] = idx_tmp[0]
        #         # idx_set[idx_tmp[2]] = idx_tmp[0]
        #         for idx, elem in enumerate(self.landmarks):
        #             self.observation[idx_set[idx]] = cal_observation(elem)
        #     else: 
        #         for idx, elem in enumerate(self.landmarks):
        #             self.observation[idx] = cal_observation(elem)      
        # else:
        #     for idx, elem in enumerate(self.landmarks):
        #         self.observation[idx] = cal_observation(elem)            

        for idx, elem in enumerate(self.landmarks):
            self.observation[idx] = cal_observation(elem)                  

        self.first_filtering = False

        # now column vector
        self.observation = self.observation.reshape((-1,1))

        return self.observation


    def generate_landmark(self):
        self.landmarks = []
        for i in range(self.num_landmark):
            self.landmarks += [(random.randint(-self.region_max, self.region_max), 
                                random.randint(-self.region_max, self.region_max))]      


###############################################################################################################
###############################################################################################################
###############################################################################################################
    def KF_net_filtering_split(self, observation):

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        self.predict_state()   

        if self.dnn_first:
            self.state_pred_past = self.state_pred.detach().clone()

        self.obs_true = torch.Tensor(observation).reshape((-1,2))

        if self.dnn_first:
            self.obs_past = self.obs_true.detach().clone()   
        
        for idx, elem in enumerate(self.landmarks):

            obs_pred = self.predict_observation(idx)
            obs_true_reshaped = self.obs_true[idx].reshape((-1,1))
            residual = obs_true_reshaped - obs_pred
                

            if self.mode in [1.3]:
                state_inno = self.prepare_input('state prediction error')
                obs_inno = residual.clone()   
                diff_state = self.prepare_input('state evolution (past)') 
                diff_obs = torch.Tensor([
                    self.obs_true[idx,0] - self.obs_past[idx,0],
                    self.obs_true[idx,1] - self.obs_past[idx,1]
                    ]).reshape((-1,1))                                                   
                # linearization_error = obs_true_reshaped - self.prepare_input('observation linearization error')
                linearization_error = obs_pred - self.prepare_input('observation linearization error')
                H_jacob = self.prepare_input('observation Jacobian')       

                if self.mode == 1.3:    
                    (Pk, Sk) = self.kf_net(state_inno, obs_inno, diff_state, diff_obs, linearization_error, H_jacob.reshape((-1,1)))     
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

            # K_gain_tmp = Pk @ torch.transpose(H_jacob, 0, 1) @ torch.linalg.inv(Sk)
            K_gain_tmp = Pk @ torch.transpose(H_jacob, 0, 1) @ Sk

            rm_idx = self.rm_idx
            K_gain = torch.zeros((3+2*self.num_landmark, 2))
            K_gain[0:3,:] = K_gain_tmp[0:3,:]
            K_gain[rm_idx:rm_idx+2,:] = K_gain_tmp[3:,:]



            ## state correction
            residual[1,0] %= 2*np.pi
            if residual[1,0] > np.pi:
                residual[1,0] -= 2*np.pi

            self.state_pred += (K_gain @ torch.Tensor(residual))  

            self.state_pred[2,0] %= 2*torch.pi
            if self.state_pred[2,0] > torch.pi:
                self.state_pred[2,0] -= 2*torch.pi      


        self.dnn_first = False
        
        self.obs_past = self.obs_true.detach().clone()       
        self.state_pred_past = self.state_pred.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.state_post = self.state_pred.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_pred.clone()), axis=1) 


    def KF_net_filtering(self, observation):
        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        self.predict_state()

        if self.dnn_first:
            self.state_pred_past = self.state_pred.detach().clone()

        self.obs_true = torch.Tensor(observation).reshape((-1,2))

        if self.dnn_first:
            self.obs_past = self.obs_true.detach().clone()   

        for idx, elem in enumerate(self.landmarks):
            obs_pred = self.predict_observation(idx)
            obs_true_reshaped = self.obs_true[idx].reshape((-1,1))
            residual = obs_true_reshaped - obs_pred
            if self.mode == 0.1:
                ## input1: x_{k-1 | k-1} - x_{k-1 | k-2}
                state_inno = self.prepare_input('state prediction error')
                ## input2: residual
                obs_inno = residual.clone()
                ## input3: x_k - x_{k-1}
                diff_state = self.prepare_input('state evolution (past)')       
                ## input4: y_k - y_{k-1}
                diff_obs = torch.Tensor([
                    self.obs_true[idx,0] - self.obs_past[idx,0],
                    self.obs_true[idx,1] - self.obs_past[idx,1]
                    ]).reshape((-1,1))    
                # K_gain 계산
                K_gain_tmp = self.kf_net(state_inno, obs_inno, diff_state, diff_obs)           

                # state_inno_in = F.normalize(state_inno, p=2, dim=0, eps=1e-12)
                # obs_inno_in = F.normalize(obs_inno, p=2, dim=0, eps=1e-12)
                # diff_state_in = F.normalize(diff_state, p=2, dim=0, eps=1e-12)
                # diff_obs_in= F.normalize(diff_obs, p=2, dim=0, eps=1e-12)
                # K_gain_tmp = self.kf_net(state_inno_in, obs_inno_in, diff_state_in, diff_obs_in)            
                                                                                  
            else:
                raise NotImplementedError()

            ## state correction
            rm_idx = self.rm_idx
            K_gain = torch.zeros((3+2*self.num_landmark, 2))
            K_gain[0:3,:] = K_gain_tmp[0:3,:]
            K_gain[rm_idx:rm_idx+2,:] = K_gain_tmp[3:,:]

            ## state correction
            residual[1,0] %= 2*np.pi
            if residual[1,0] > np.pi:
                residual[1,0] -= 2*np.pi

            self.state_pred += (K_gain @ torch.Tensor(residual))            

            self.state_pred[2,0] %= 2*torch.pi
            if self.state_pred[2,0] > torch.pi:
                self.state_pred[2,0] -= 2*torch.pi  


        self.dnn_first = False

        self.state_pred_past = self.state_pred.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = self.obs_true.detach().clone()        
        self.state_post = self.state_pred.detach().clone()
        self.state_history = torch.cat((self.state_history, self.state_pred.clone()), axis=1)

    def predict_state(self):
        dtheta = self.dtheta
        angle_post_est = self.state_post[2][0]
        Fx = torch.cat(
            (torch.eye(3), torch.zeros((3,2*self.num_landmark))), 
            axis=1)
        dx = torch.Tensor([torch.cos(angle_post_est)*self.velocity, torch.sin(angle_post_est)*self.velocity, dtheta]) \
                        .reshape((-1,1))

        self.state_pred = self.state_post + torch.transpose(Fx, 0, 1) @ dx
        self.state_pred[2,0] %= 2*torch.pi
        if self.state_pred[2,0] > torch.pi:
            self.state_pred[2,0] -= 2*torch.pi     

    def predict_observation(self, idx):
        self.x_pred = self.state_pred[0,0]
        self.y_pred = self.state_pred[1,0]
        self.angle_pred = self.state_pred[2,0]            
        x_pred = self.x_pred
        y_pred = self.y_pred
        angle_pred = self.angle_pred        
        
        ## observation prediction
        self.rm_idx = 3 + idx * 2
        rm_idx = self.rm_idx
        range = self.obs_true[idx, 0]
        bearing = self.obs_true[idx, 1]

        if not self.landmarks_visited[idx]:
            rm_x = x_pred + range * torch.cos(bearing + angle_pred)
            rm_y = y_pred + range * torch.sin(bearing + angle_pred)

            self.state_pred[rm_idx] = rm_x
            self.state_pred[rm_idx+1] = rm_y

            self.landmarks_visited[idx] = 1

        self.dx_lm_robot = self.state_pred[rm_idx, 0] - x_pred
        self.dy_lm_robot = self.state_pred[rm_idx+1, 0] - y_pred
        dx_lm_robot = self.dx_lm_robot
        dy_lm_robot = self.dy_lm_robot

        q = dx_lm_robot**2 + dy_lm_robot**2        

        # estimated observation
        obs_pred = torch.Tensor([torch.sqrt(q).item(), (torch.atan2(dy_lm_robot,dx_lm_robot)-angle_pred).item()]) \
                    .reshape((-1,1))        

        return (obs_pred)

    def prepare_input(self, mode=None):
        mode_valid = ['state prediction error and dlm_robot',
                        'state prediction error',
                        'state evolution (past)',
                        'observation evolution',
                        'observation linearization error',
                        'observation Jacobian']
        if mode not in mode_valid:
            raise NotImplementedError()

        with torch.no_grad():     

            if mode == 'state prediction error and dlm_robot':
                x_post_past = self.state_post_past[0,0]
                y_post_past = self.state_post_past[1,0]
                angle_post_past = self.state_post_past[2,0]

                x_pred_past = self.state_pred_past[0,0]
                y_pred_past = self.state_pred_past[1,0]
                angle_pred_past = self.state_pred_past[2,0]
        
                state_inno = torch.Tensor([
                    x_post_past-x_pred_past,
                    y_post_past-y_pred_past,
                    angle_post_past-angle_pred_past,
                    self.dx_lm_robot,
                    self.dy_lm_robot]
                    ).reshape((-1,1)) 
                return state_inno   

            if mode == 'state prediction error':
                rm_idx = self.rm_idx

                x_post_past = self.state_post_past[0,0]
                y_post_past = self.state_post_past[1,0]
                angle_post_past = self.state_post_past[2,0]
                dx_post_past = self.state_post_past[rm_idx, 0]
                dy_post_past = self.state_post_past[rm_idx+1,0]

                x_pred_past = self.state_pred_past[0,0]
                y_pred_past = self.state_pred_past[1,0]
                angle_pred_past = self.state_pred_past[2,0]
                dx_pred_past = self.state_pred_past[rm_idx, 0]
                dy_pred_past = self.state_pred_past[rm_idx+1, 0]
        
                state_inno = torch.Tensor([
                    x_post_past-x_pred_past, 
                    y_post_past-y_pred_past,
                    angle_post_past-angle_pred_past,
                    dx_post_past-dx_pred_past, 
                    dy_post_past-dy_pred_past
                ]).reshape((-1,1))
                return state_inno

            if mode == 'state evolution (past)':
                rm_idx = self.rm_idx
                diff_state = torch.Tensor([
                    self.state_post[0,0] - self.state_post_past[0,0],
                    self.state_post[1,0] - self.state_post_past[1,0],
                    self.state_post[2,0] - self.state_post_past[2,0],
                    self.state_post[rm_idx,0] - self.state_post_past[rm_idx,0],
                    self.state_post[rm_idx+1,0] - self.state_post_past[rm_idx+1,0]]
                    ).reshape((-1,1))           
                return diff_state   


            if mode == 'observation linearization error':
                dx_lm_robot = self.dx_lm_robot
                dy_lm_robot = self.dy_lm_robot
                rm_idx = self.rm_idx
                state_predict = torch.Tensor([
                    self.x_pred, 
                    self.y_pred, 
                    self.angle_pred, 
                    self.state_pred[rm_idx, 0], 
                    self.state_pred[rm_idx+1, 0] ]
                    ).reshape((-1,1))    

                q = dx_lm_robot**2 + dy_lm_robot**2  + self.eps            

                H_jacob = torch.Tensor(
                            [[-torch.sqrt(q)*dx_lm_robot, -torch.sqrt(q)*dy_lm_robot, 0, torch.sqrt(q)*dx_lm_robot, torch.sqrt(q)*dy_lm_robot],
                            [dy_lm_robot, -dx_lm_robot, -q, -dy_lm_robot, dx_lm_robot]]) / q
                return (H_jacob @ state_predict) 

            if mode == 'observation Jacobian':
                dx_lm_robot = self.dx_lm_robot
                dy_lm_robot = self.dy_lm_robot
                q = dx_lm_robot**2 + dy_lm_robot**2  + self.eps              

                H_jacob = torch.Tensor(
                            [[-torch.sqrt(q)*dx_lm_robot, -torch.sqrt(q)*dy_lm_robot, 0, torch.sqrt(q)*dx_lm_robot, torch.sqrt(q)*dy_lm_robot],
                            [dy_lm_robot, -dx_lm_robot, -q, -dy_lm_robot, dx_lm_robot]]) / q       
                return H_jacob      
 


    def EKF_filtering(self, observation):
        def predict_state(self):
            angle_post_est = self.state_post[2][0]
            Fx = np.concatenate(
                (np.eye(3), np.zeros((3,2*self.num_landmark))), 
                axis=1)
            dx = np.array([np.cos(angle_post_est)*self.velocity, np.sin(angle_post_est)*self.velocity, dtheta]) \
                            .reshape((-1,1))

            self.state_pred = self.state_post + np.transpose(Fx) @ dx
            self.state_pred[2,0] %= 2*np.pi
            if self.state_pred[2,0] > np.pi:
                self.state_pred[2,0] -= 2*np.pi      

        def angle_modulo(val):
            val %= 2*np.pi
            if val > np.pi:
                val -= 2*np.pi     
            return val      
            
                
        dtheta = self.dtheta

        ## state prediction
        predict_state(self)
        
        ## Cov prediction
        angle_post_est = self.state_post[2][0]        
        F_jacob = np.eye(3 + 2*self.num_landmark)
        F_tmp = np.array([[1, 0, -np.sin(angle_post_est)], 
                        [0, 1, np.cos(angle_post_est)],
                        [0, 0, 1]])
        F_jacob[0:3,0:3] = F_tmp
        self.cov_pred = (F_jacob @ self.cov_post @ np.transpose(F_jacob)
                            + self.cov_processing)


        ## Correction
        obs_true = observation.reshape((-1,2))

        self.state_pred_corrected = self.state_pred.copy()
        for idx, elem in enumerate(self.landmarks):
            
            x_pred = self.state_pred_corrected[0][0]
            y_pred = self.state_pred_corrected[1][0]
            angle_pred = self.state_pred_corrected[2][0]

            ## observation prediction
            rm_idx = 3 + idx * 2
            range_ = obs_true[idx][0]
            bearing = obs_true[idx][1]

            if not self.landmarks_visited[idx]:
                rm_x = x_pred + range_ * np.cos(bearing+angle_pred)
                rm_y = y_pred + range_ * np.sin(bearing+angle_pred)

                self.state_pred_corrected[rm_idx] = rm_x
                self.state_pred_corrected[rm_idx+1] = rm_y

                self.landmarks_visited[idx] = 1
                
            dx_lm_robot = self.state_pred_corrected[rm_idx][0] - x_pred
            dy_lm_robot = self.state_pred_corrected[rm_idx+1][0] - y_pred

            q = dx_lm_robot**2 + dy_lm_robot**2 + self.eps
            obs_pred = np.array([np.sqrt(q).item(), (np.arctan2(dy_lm_robot,dx_lm_robot)-angle_pred).item()]) \
                        .reshape((-1,1))

            F_xj = np.zeros((5, 3+2*self.num_landmark))
            F_xj[0:3,0:3] = np.eye(3)
            F_xj[3:,rm_idx:rm_idx+2] = np.eye(2)
            H_jacob = np.array(
                        [[-np.sqrt(q)*dx_lm_robot, -np.sqrt(q)*dy_lm_robot, 0, np.sqrt(q)*dx_lm_robot, np.sqrt(q)*dy_lm_robot],
                        [dy_lm_robot, -dx_lm_robot, -q, -dy_lm_robot, dx_lm_robot]])
            H_jacob = H_jacob / q @ F_xj


            K_gain = self.cov_pred @ np.transpose(H_jacob) @ \
                np.linalg.inv(H_jacob@self.cov_pred@np.transpose(H_jacob)+self.cov_observation)
            
            ## state correction
            obs_true_reshaped = obs_true[idx].reshape((-1,1))
            residual = obs_true_reshaped - obs_pred

            residual[1,0] = angle_modulo(residual[1,0])

            self.state_pred_corrected = self.state_pred_corrected + (K_gain @ residual)

            self.state_pred_corrected[2,0] = angle_modulo(self.state_pred_corrected[2,0])

            self.cov_pred = ( np.eye(3+2*self.num_landmark) - K_gain @ H_jacob ) @ self.cov_pred


        self.state_post = self.state_pred_corrected.copy()
        self.cov_post = self.cov_pred       

        self.state_history = np.concatenate((self.state_history, self.state_post), axis=1)






