import enum
from .robot import Robot
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

class RTplot():
    def __init__(self, robot_list, color_list, label_list, iter_max, path):
        self.robot_list = robot_list
        self.robot = robot_list[0]
        self.num_robot = len(robot_list)
        
        self.color_list = color_list
        self.label_list = label_list

        self.iter_max = iter_max
        self.save_path = path

        self.X = []
        self.Y = []
        self.X_hat_list = []
        self.Y_hat_list = []
        self.state_tensor_list = []

        self.X += [self.robot.X[0]]
        self.Y += [self.robot.Y[0]]
        for robot in robot_list:
            self.X_hat_list.append([robot.X[0]])
            self.Y_hat_list.append([robot.Y[0]])
            if robot.is_dnn:
                self.state_tensor_list.append([torch.zeros((3+2*self.robot.num_landmark, 0))])
            else:
                self.state_tensor_list.append([np.zeros((3+2*self.robot.num_landmark, 0))])


        self.upCount = 0
        self.fig = plt.figure(figsize=(6,6))
        self.ax = plt.subplot()
        plt.autoscale(enable=True)

    def aniFunc(self, i):
        if i == 0:
            return 

        if self.upCount > 20:
            self.clearPoint()

        if i % 10 == 0:
            print(i)

        # Robot control.. if this RT plot is on
        x, y, angle = self.robot.move()
        
        for robot in self.robot_list:
            robot.dtheta = self.robot.dtheta
            robot.X += [x]
            robot.Y += [y]
            robot.Angle += [angle]

        self.X.append(x)
        self.Y.append(y)
        self.upCount += 1

        # Robot observation
        observation = self.robot.observe()

        # EKF filtering
        with torch.no_grad():
            for idx, robot in enumerate(self.robot_list):
                robot.estimate_state(observation)
                if robot.is_dnn:
                    self.X_hat_list[idx].append(robot.state_post[0,0].detach())
                    self.Y_hat_list[idx].append(robot.state_post[1,0].detach())
                else:
                    self.X_hat_list[idx].append(robot.state_post[0,0])
                    self.Y_hat_list[idx].append(robot.state_post[1,0])       


        # Drawing (True map)
        self.ax.cla()
        lms = self.robot.landmarks
        self.ax.scatter(
            list(map(lambda x: x[0], lms)),
            list(map(lambda x: x[1], lms)), 
            s=72,
            c='gray',
            marker='X'
        )

        angle_radian = angle
        self.ax.plot(self.X, self.Y, linewidth=2.0, color='gray',
                        marker='o', markersize=10, markevery=[-1],
                        label='Ground truth')        
        self.ax.quiver(x, y, np.cos(angle_radian), np.sin(angle_radian), color='gray')

        # Drawing (for all robot in robot_list, draw estimated map)
        for idx, robot in enumerate(self.robot_list):
            if robot.is_dnn:
                angle_hat = robot.state_post[2,0].detach()
                lms_hat = list(robot.state_post[3:,0].detach())
            else:
                angle_hat = robot.state_post[2,0]
                lms_hat = list(robot.state_post[3:,0])
            
            lms_x_hat = lms_hat[::2]
            lms_y_hat = lms_hat[1::2]

            self.ax.scatter(lms_x_hat, lms_y_hat, 
                s=72, c=self.color_list[idx], marker='P')

            self.ax.plot(self.X_hat_list[idx], self.Y_hat_list[idx], 
                linewidth=2.0, linestyle='dashed', color=self.color_list[idx],
                label=self.label_list[idx])

            self.ax.quiver(self.X_hat_list[idx][-1], self.Y_hat_list[idx][-1], 
                np.cos(angle_hat), np.sin(angle_hat), 
                color=self.color_list[idx])

        
        self.ax.legend(loc='lower right', fontsize=12)
        if self.robot.is_test:
            # a = 250
            a = 40
            # self.ax.set(xlim=(-40,40), ylim=(-40,40))
            # self.ax.set(xlim=(-120,120), ylim=(-120,120))
            self.ax.set(xlim=(-a,a), ylim=(-a,a))
        else:
            a = 40
            # self.ax.set(xlim=(-22,22), ylim=(-22,22))
            # self.ax.set(xlim=(-60,60), ylim=(-60,60))
            self.ax.set(xlim=(-a,a), ylim=(-a,a))
        self.ax.grid(True)
        self.ax.set_xlabel('x [m]', fontsize=14)
        self.ax.set_ylabel('y [m]', fontsize=14)



    def clearPoint(self):
        self.X = self.X[1:]
        self.Y = self.Y[1:]        
        for idx, elem in enumerate(self.robot_list):
            self.X_hat_list[idx] = self.X_hat_list[idx][1:]
            self.Y_hat_list[idx] = self.Y_hat_list[idx][1:]
    
    
    def start(self):
        self.ani = FuncAnimation(self.fig, self.aniFunc, 
                        frames=np.arange(0,self.iter_max,1),
                        interval=500,
                        repeat=False)

        # plt.show()

        
    def save_video(self):
        writer = writers['ffmpeg']
        writer = writer(fps=10, metadata={'artist': 'Me'}, bitrate=1800)

        self.ani.save(self.save_path, writer)
        self.fig.savefig('map.eps', format='eps')
        print('Saved!')
