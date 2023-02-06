import torch
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

n = 0

fig, ax = plt.subplots()

data_x = torch.load('./.data/NCLT/test/KF v1 x_hat.pt')
x = data_x[n][0][1:].numpy()
y = data_x[n][3][1:].numpy()
ax.plot(x, y, 
    label='KalmanNet',
    linewidth=2, color='#0000ff', linestyle='dashdot')


data_x = torch.load('./.data/NCLT/test/SKF x_hat.pt')
x = data_x[n][0][1:].numpy()
y = data_x[n][3][1:].numpy()
ax.plot(x, y, 
    label='Split-KalmanNet',
    linewidth=2, color='#ff0000', linestyle='solid')

data_x = torch.load('./.data/NCLT/test/EKF x_hat.pt')
x = data_x[n][0][1:].numpy()
y = data_x[n][3][1:].numpy()
ax.plot(x, y, 
    label='EKF (mismatch)',
    linewidth=2, color='#00ff00', linestyle='dotted')

data_x = torch.load('./.data/NCLT/test/state.pt')
print(data_x.shape)
x_true = data_x[n][0][1:].numpy()
y_true = data_x[n][3][1:].numpy()
ax.plot(x_true, y_true, 
    label='Ground truth',
    linewidth=2, color='#808080', linestyle='solid'
    )



# data_x = torch.load('./.data/NCLT/test/KF v2 x_hat.pt')
# x = data_x[n][0][1:].numpy()
# y = data_x[n][3][1:].numpy()
# ax.plot(x, y, label='KalmanNet 2')




ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('x [m]', fontsize=14, fontweight='bold')
ax.set_ylabel('y [m]', fontsize=14, fontweight='bold')

ax.legend(fontsize=14)
ax.grid()
fig.savefig('nclt_trajectory.eps', format='eps')

plt.show()
