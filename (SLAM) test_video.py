from EKFSLAM.robot import Robot
from EKFSLAM.render import RTplot
import torch


is_test = True
is_mismatch = False


robot0 = Robot(is_dnn=False, is_test=is_test, is_mismatch=is_mismatch, mode=0)
robot01 = Robot(is_dnn=False, is_test=is_test, is_mismatch=is_mismatch, mode=1)

robot1 = Robot(is_dnn=True, is_test=is_test, mode=0.1, is_mismatch=is_mismatch)
robot1.kf_net = torch.load('./EKFSLAM/SLAM_model_trained/KalmanNet_1000.pt')
robot1.kf_net.initialize_hidden()

robot2 = Robot(is_dnn=True, is_test=is_test, mode=1.3, is_mismatch=is_mismatch)
robot2.kf_net = torch.load('./EKFSLAM/SLAM_model_trained/Split_KalmanNet_1000.pt')
robot2.kf_net.initialize_hidden()


robot_list = [robot0, robot01, robot1, robot2]
color_list = ['#000000', '#ff0000', '#0000ff', '#77ac30']
label_list = ['EKF (perfect)', 'EKF (mismatch)', 'KalmanNet', 'Split-KalmanNet']

plot_handler = RTplot(
                robot_list=robot_list,
                color_list=color_list,
                label_list=label_list,
                iter_max=500,
                path='test_video.mp4')

plot_handler.start()
plot_handler.save_video()      