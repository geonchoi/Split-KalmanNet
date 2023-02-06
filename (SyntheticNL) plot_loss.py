import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

fontsize = 22
labelsize = fontsize
linewidth = 3

x = np.arange(10, 501, 10)
selection = np.arange(0, x.shape[0], 3)
selection = np.arange(0, x.shape[0], 1)

fig, ax = plt.subplots()

loss_val = np.load('./valid_loss_kalman_nonlinear.npy')
print(loss_val.shape)
ax.plot(x[selection], loss_val[selection], 
    label='KalmanNet',
    linewidth=linewidth, color='#0000ff', linestyle='dashdot')

loss_val = np.load('./valid_loss_split_nonlinear.npy')
print(loss_val.shape)
ax.plot(x[selection], loss_val[selection], 
    label='Split-KalmanNet',
    linewidth=linewidth, color='#ff0000', linestyle='solid')

loss_val = np.load('./valid_loss_ekf_nonlinear.npy')
print(loss_val.shape)
ax.plot(x[selection], np.ones(x[selection].shape) * loss_val,
    label='EKF (perfect)',
    linewidth=linewidth, color='#000000', linestyle='dashed')

ax.xaxis.set_tick_params(labelsize=labelsize)
ax.yaxis.set_tick_params(labelsize=labelsize)
ax.set_xlabel('Training round', fontsize=fontsize, fontweight='bold')
ax.set_ylabel('Loss [dB]', fontsize=fontsize, fontweight='bold')

ax.legend(fontsize=fontsize)
ax.grid()
plt.tight_layout()
fig.savefig('convergence_nonlinear.eps', format='eps')

plt.show()
