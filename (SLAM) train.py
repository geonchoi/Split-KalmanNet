###################
### Train model ###
###################

from EKFSLAM.robot import Robot
from EKFSLAM.trainer import Trainer_SLAM

max_iter = 1000
is_mismatch = False
seq_len = 10

# S_KalmanNet   

robot = Robot(is_dnn=True, is_test=False, mode=1.3, is_mismatch=is_mismatch)
trainer = Trainer_SLAM(robot=robot, path='Split_KalmanNet.pt')
trainer.seq_len = seq_len
for i in range(max_iter):
    trainer.train_batch()
    trainer.robot.reset(clean_history=True)            
       

robot = Robot(is_dnn=True, is_test=False, mode=0.1, is_mismatch=is_mismatch)
trainer = Trainer_SLAM(robot=robot, path='KalmanNet.pt')
trainer.seq_len = seq_len
for i in range(max_iter):
    trainer.train_batch()
    trainer.robot.reset(clean_history=True)            