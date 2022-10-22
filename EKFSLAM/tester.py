import torch
import os
from EKFSLAM.robot import Robot
import numpy as np

print_period = 200

class Tester():

    def __init__(self, robot_list, name_list, is_long_seq=False):
        self.robot_list = robot_list
        self.name_list = name_list

        self.loss_fn_test = torch.nn.MSELoss()
        self.loss_list = []
        self.loss_last_list = []
        
        if not is_long_seq:
        # for not is_test (short seq)            
            self.seq_len = 20
            self.test_num = 100
        else:
        # for is_test (long seq)
            # self.seq_len = 200
            # self.test_num = 100
            self.seq_len = 50
            self.test_num = 100

        self.path_workspace = os.getcwd()
        model_all = os.listdir()
        model_list = []
        for elem in self.name_list:
            model_list += [list(filter(
                lambda model: elem in model, model_all
            ))]

        epoch_list = []
        for idx, elem in enumerate(self.name_list):
            epoch_list += [list(map(
                lambda model: int(model[:-3].split('_')[-1]), model_list[idx]
            ))]
            epoch_list[-1].sort()
            ## choose 500의 배수
            # epoch_list[-1] = epoch_list[-1][0::2]
            ## Choose last one only
            epoch_list[-1] = [epoch_list[-1][-1]]
        self.epoch_list = epoch_list

        self.test_model_list = []
        for idx, elem in enumerate(self.epoch_list):
            self.test_model_list += [list(map(
                lambda num: self.name_list[idx] + '_' + str(num) + '.pt', elem
            ))]

    def test_equal(self):
        loss_list = []
        loss_last_list = []

        # generate test data
        self.generate_data()

        # EKF (exact)
        loss_list += [[]]
        loss_last_list += [[]]
        robot = Robot(is_dnn=False, is_test=self.robot_list[0].is_test, is_mismatch=self.robot_list[0].is_mismatch, mode=0)

        (loss_tmp, loss_last_tmp) = self.test_loss_equal(robot)
        loss_list[-1] += [loss_tmp]
        loss_last_list[-1] += [loss_last_tmp]      

        # EKF (mismatch)
        if not self.robot_list[0].is_mismatch:
            loss_list += [[]]
            loss_last_list += [[]]
            robot = Robot(is_dnn=False, is_test=self.robot_list[0].is_test, is_mismatch=self.robot_list[0].is_mismatch, mode=1)

            (loss_tmp, loss_last_tmp) = self.test_loss_equal(robot)
            loss_list[-1] += [loss_tmp]
            loss_last_list[-1] += [loss_last_tmp]      

        ## DNN
        for idx, robot in enumerate(self.robot_list):
            loss_list += [[]]
            loss_last_list += [[]]
            for test_path in self.test_model_list[idx]:
                model_path = self.path_workspace + '/' + test_path
                print(model_path)
                robot.kf_net = torch.load(model_path).eval()
                robot.kf_net.initialize_hidden()

                (loss_tmp, loss_last_tmp) = self.test_loss_equal(robot)
                loss_list[-1] += [loss_tmp]
                loss_last_list[-1] += [loss_last_tmp]


        ## loss list postprocessing
        self.loss_list = np.array(self.loss_list)
        self.loss_last_list = np.array(self.loss_last_list)
        self.loss_mean_dB = 10*np.log10(np.mean(self.loss_list, axis=1))
        self.loss_std_dB = 10*np.log10(np.std(self.loss_list, axis=1))
        self.loss_last_mean_dB = 10*np.log10(np.mean(self.loss_last_list, axis=1))
        self.loss_last_std_dB = 10*np.log10(np.std(self.loss_last_list, axis=1))        
 
        return (loss_list, loss_last_list)


    def generate_data(self):
        robot_tmp = self.robot_list[0]
        self.test_robot = Robot(is_dnn=False, is_test=self.robot_list[0].is_test, is_mismatch=self.robot_list[0].is_mismatch)
        with torch.no_grad():
            robot = self.test_robot
            self.test_data = torch.zeros((3+2*robot.num_landmark+1, self.seq_len, self.test_num)) # 마지막 하나는 dtheta 자리
            self.test_observation = torch.zeros((2*robot.num_landmark, self.seq_len, self.test_num))
            for i in range(self.test_num):
                state_tensor = torch.zeros((3+2*robot.num_landmark+1, 0))
                dtheta_list = [0]
                for ii in range(self.seq_len):
                    robot.move()
                    dtheta_list += [robot.dtheta]
                    self.test_observation[:,ii,i] = torch.tensor(robot.observe().reshape(-1))

                rm_x_list = list(map(lambda x: x[0], robot.landmarks))
                rm_y_list = list(map(lambda x: x[1], robot.landmarks))

                state_list = [robot.X, robot.Y, robot.Angle]
                seq_len = len(robot.X)
                for ii in range(robot.num_landmark):
                    state_list += [[rm_x_list[ii]] * seq_len]
                    state_list += [[rm_y_list[ii]] * seq_len]
                state_list += [dtheta_list]

                state_tensor = torch.cat((state_tensor, torch.Tensor(state_list)), dim=1)

                state_tensor = state_tensor[:,1:]
                self.test_data[:,:,i] = state_tensor.clone()

                robot.reset(clean_history=True)

    def test_loss_equal(self, robot):
        loss_list = []
        loss_last_list = []
        if robot.is_dnn:
            loss_sum = 0
            loss_last_sum = 0
            with torch.no_grad():
                for i in range(self.test_num):

                    if i % print_period == 0: 
                        print(f'{i} / {self.test_num}')

                    state_tensor = self.test_data[:,:,i]
                    for ii in range(self.seq_len):
                        robot.dtheta = state_tensor[-1,ii]
                        observation = self.test_observation[:,ii,i].reshape((-1,1))
                        robot.estimate_state(observation)
                
                    history_tensor = robot.state_history[:,1:]
                    loss_sum += self.loss_fn_test(state_tensor[:-1,:], history_tensor).item()

                    # state_tensor_end = state_tensor[:-1,-1]
                    # history_tensor_end = history_tensor[:,-1]
                    state_tensor_end = state_tensor[:-1,-25:]
                    history_tensor_end = history_tensor[:,-25:]                       
                    loss_last_sum += self.loss_fn_test(state_tensor_end, history_tensor_end).item()
                    
                    loss_list += [self.loss_fn_test(state_tensor[:-1,:], history_tensor).item()]
                    loss_last_list += [self.loss_fn_test(state_tensor_end, history_tensor_end).item()]

                    robot.reset(clean_history=True)
        else:
            loss_sum = 0
            loss_last_sum = 0

            for i in range(self.test_num):

                if i % print_period == 0: 
                    print(f'{i} / {self.test_num}')

                state_tensor = self.test_data[:,:,i]
                for ii in range(self.seq_len):
                    robot.dtheta = state_tensor[-1,ii].numpy()
                    observation = self.test_observation[:,ii,i].reshape((-1,1)).numpy()
                    robot.estimate_state(observation)
            
                history_tensor = robot.state_history[:,1:]
                loss_sum += self.loss_fn_test(state_tensor[:-1,:], torch.Tensor(history_tensor)).item()

                # state_tensor_end = state_tensor[:-1,-1]
                # history_tensor_end = history_tensor[:,-1]
                state_tensor_end = state_tensor[:-1,-25:]
                history_tensor_end = history_tensor[:,-25:]                
                loss_last_sum += self.loss_fn_test(state_tensor_end, torch.Tensor(history_tensor_end)).item()
                
                loss_list += [self.loss_fn_test(state_tensor[:-1,:], torch.Tensor(history_tensor)).item()]
                loss_last_list += [self.loss_fn_test(state_tensor_end, torch.Tensor(history_tensor_end)).item()]

                robot.reset(clean_history=True)                

        self.loss_list += [loss_list]
        self.loss_last_list += [loss_last_list]


        loss_dB = 10*torch.log10(torch.tensor(loss_sum / self.test_num))
        loss_dB_last = 10*torch.log10(torch.tensor(loss_last_sum / self.test_num))
        print(f'loss [dB] = {loss_dB:.4f}, loss last [dB] = {loss_dB_last:.4f}')
        return (loss_sum / self.test_num, loss_last_sum / self.test_num)
