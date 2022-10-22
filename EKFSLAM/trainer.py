from .robot import Robot
import torch


# Learning params for Split architecture
lr_s = 1e-3 
wd_s = 0
# lr_s = 1e-4
# wd_s = 1e-4

# Learning params for joint architecture
# lr_k = 1e-4
# wd_k = 1e-4
lr_k = 1e-3
wd_k = 0

# lr_k = 5e-4
# lr_s = 5e-2

save_period = 100

class Trainer_SLAM():

    def __init__(self, robot:Robot, path):
        self.robot = robot
        self.path = path
        self.mode = self.robot.mode

        self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss_fn_test = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(robot.kf_net.parameters(), lr=lr_k, weight_decay=wd_k)

        if self.mode in [1.3]:
            self.optim_for_split()                  

        cal_num_param = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(cal_num_param(robot.kf_net))

        # self.batch_size = 8
        self.batch_size = 1
        self.alter_num = 3
        self.train_count = 0


    def train_batch(self):
        if self.mode in [0.1]:
            self.train_batch_joint()
        elif self.mode in [1.3]:
            self.train_batch_alternative()
        else:
            raise NotImplementedError()

    def train_batch_alternative(self):

        if self.train_count > 0 and self.train_count % self.alter_num == 0:

            # self.optimizer_list = [torch.optim.Adam(self.param_group_1, lr=lr_s, weight_decay=wd_s),
                                    # torch.optim.Adam(self.param_group_2, lr=lr_s, weight_decay=wd_s)]

            if self.unfreeze_net_current == 1:
                # unfreeze only 2             
                for elem in self.network1:
                    for param in elem.parameters():
                        param.requires_grad = False
                for elem in self.network2:
                    for param in elem.parameters():
                        param.requires_grad = True             
                self.unfreeze_net_current = 2
            elif self.unfreeze_net_current == 2:
                # unfreeze only 1
                for elem in self.network1:
                    for param in elem.parameters():
                        param.requires_grad = True
                for elem in self.network2:
                    for param in elem.parameters():
                        param.requires_grad = False            
                self.unfreeze_net_current = 1
        
        self.optimizer = self.optimizer_list[self.unfreeze_net_current-1]     

        self.optimizer.zero_grad()
        state_tensor = torch.zeros((3+2*self.robot.num_landmark, 0))
        for i in range(self.batch_size):
            for ii in range(self.seq_len):
                self.robot.move()
                observation = self.robot.observe()
                self.robot.estimate_state(observation)
            
            rm_x_list = list(map(lambda x: x[0], self.robot.landmarks))
            rm_y_list = list(map(lambda x: x[1], self.robot.landmarks))

            state_list = [self.robot.X, self.robot.Y, self.robot.Angle]
            seq_len = len(self.robot.X)
            for ii in range(self.robot.num_landmark):
                state_list += [[rm_x_list[ii]] * seq_len]
                state_list += [[rm_y_list[ii]] * seq_len]

            state_tensor = torch.cat((state_tensor, torch.Tensor(state_list)), dim=1)
            self.robot.reset()


        state_tensor = state_tensor[:,1:]
        history_tensor = self.robot.state_history[:, 1:-1]

        loss = self.loss_fn(state_tensor, history_tensor)
        loss.backward()

        ## gradient clipping with maximum value 5
        torch.nn.utils.clip_grad_norm_(self.robot.kf_net.parameters(), 1)

        # for param in self.robot.kf_net.parameters():
        #     print (self.train_count, param.data)
        #     break
        # self.optimizer.step()
        # for param in self.robot.kf_net.parameters():
        #     print (self.train_count, param.data)
        #     break

        self.optimizer.step()

        # if self.mode == 1.3:
        #     self.scheduler_list[self.unfreeze_net_current-1].step()   

        ## post-processing
        self.train_count += 1
        if self.train_count % save_period == 0:
            try:
                torch.save(self.robot.kf_net, './EKFSLAM/SLAM_model_trained/' + self.path[:-3] + '_' + str(self.train_count) + '.pt')  
            except:
                print('here')
                pass
        if self.train_count % 10 == 1:
            print(f'[Model {self.path}] [Train {self.train_count}] loss = {loss:.4f}')



    def train_batch_joint(self):
        self.optimizer.zero_grad()
        state_tensor = torch.zeros((3+2*self.robot.num_landmark, 0))
        for i in range(self.batch_size):
            for ii in range(self.seq_len):
                self.robot.move()
                observation = self.robot.observe()
                self.robot.estimate_state(observation)
            
            rm_x_list = list(map(lambda x: x[0], self.robot.landmarks))
            rm_y_list = list(map(lambda x: x[1], self.robot.landmarks))

            state_list = [self.robot.X, self.robot.Y, self.robot.Angle]
            seq_len = len(self.robot.X)
            for ii in range(self.robot.num_landmark):
                state_list += [[rm_x_list[ii]] * seq_len]
                state_list += [[rm_y_list[ii]] * seq_len]

            state_tensor = torch.cat((state_tensor, torch.Tensor(state_list)), dim=1)
            self.robot.reset()


        state_tensor = state_tensor[:,1:]
        history_tensor = self.robot.state_history[:, 1:-1]

        loss = self.loss_fn(state_tensor, history_tensor)
        loss.backward()

        ## gradient clipping with maximum value 5
        torch.nn.utils.clip_grad_norm_(self.robot.kf_net.parameters(), 1)

        self.optimizer.step()
        self.train_count += 1


        if self.train_count % save_period == 0:
            try:
                torch.save(self.robot.kf_net, './EKFSLAM/SLAM_model_trained/' + self.path[:-3] + '_' + str(self.train_count) + '.pt')  
            except:
                print('here')
                pass
        if self.train_count % 10 == 1:
            print(f'[Model {self.path}] [Train {self.train_count}] loss = {loss:.4f}')


    def optim_for_split(self):
        if self.mode in [1.3]:
            # network 1 = l1, GRU1, l2
            self.network1 = [self.robot.kf_net.l1, self.robot.kf_net.GRU1, self.robot.kf_net.l2]
            # network 2 = l3, GRU2, l4
            self.network2 = [self.robot.kf_net.l3, self.robot.kf_net.GRU2, self.robot.kf_net.l4]
            self.param_group_1 = []
            for elem in self.network1:
                self.param_group_1 += [{'params': elem.parameters()}]
            self.param_group_2 = []
            for elem in self.network2:
                self.param_group_2 += [{'params': elem.parameters()}]


            # self.optimizer_list = [torch.optim.SGD(self.param_group_1, lr=lr_s, momentum=0.8),
            #                     torch.optim.SGD(self.param_group_2, lr=lr_s, momentum=0.8)]
            # self.scheduler_list = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_list[0], T_0=4000, T_mult=100, eta_min=5e-4),
            #                     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_list[1], T_0=4000, T_mult=100, eta_min=5e-4),]

            self.optimizer_list = [torch.optim.Adam(self.param_group_1, lr=lr_s, weight_decay=wd_s),
                                    torch.optim.Adam(self.param_group_2, lr=lr_s, weight_decay=wd_s)]
                                    
            for elem in self.network1:
                for param in elem.parameters():
                    param.requires_grad = True
            for elem in self.network2:
                for param in elem.parameters():
                    param.requires_grad = False                
            self.unfreeze_net_current = 1  
    
        else:
            raise NotImplementedError()