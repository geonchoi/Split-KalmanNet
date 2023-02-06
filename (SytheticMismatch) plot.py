import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

selection = np.arange(0, 999, 3)

fig, ax = plt.subplots()

# loss_val = np.load('./test_loss_kalman_slow_linear.npy')
# print(loss_val.shape)
# ax.plot(selection, loss_val[selection],
#     label='KalmanNet',
#     linewidth=2, color='#0000ff', linestyle='dashdot')

loss_val = np.load('./test_loss_split_slow_linear.npy')
print(loss_val.shape)
ax.plot(selection, loss_val[selection],
    label='Split-KalmanNet',
    linewidth=2, color='#ff0000', linestyle='solid')

loss_val = np.load('./test_loss_ekf_slow_linear.npy')
print(loss_val.shape)
ax.plot(selection, loss_val[selection],
    label='EKF (perfect)',
    linewidth=2, color='#000000', linestyle='dashed')

loss_val = np.load('./test_loss_ekf_mismatch_slow_linear.npy')
print(loss_val.shape)
ax.plot(selection, loss_val[selection],
    label='EKF (mismatch)',
    linewidth=2, color='#00ff00', linestyle='dashed')

ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel('Time [sample]', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss [dB]', fontsize=14, fontweight='bold')

ax.legend(fontsize=12, loc='lower left')
ax.grid()
fig.savefig('timevarying.eps', format='eps')

plt.show()
