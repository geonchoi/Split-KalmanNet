import scipy.io as sio
import torch

# mode = 'train'
# mode = 'valid'
mode = 'test'

###### Training data ######
if mode == 'train':
    data_x = sio.loadmat('data_x.mat')['data_x']
    data_y = sio.loadmat('data_y.mat')['data_y']


    train_x = torch.tensor(data_x[0:80, [0,2,4,1,3,5]], dtype=torch.float32)
    train_y = torch.tensor(data_y[0:80], dtype=torch.float32)
    torch.save(train_x, './.data/NCLT/train/state.pt')
    torch.save(train_y, './.data/NCLT/train/obs.pt')
    print(train_x.shape)
    print(train_y.shape)

###### Validation data ######
if mode == 'valid':
    data_x = sio.loadmat('data_x.mat')['data_x']
    data_y = sio.loadmat('data_y.mat')['data_y']


    valid_x = torch.tensor(data_x[20:25, [0,2,4,1,3,5]], dtype=torch.float32)
    valid_y = torch.tensor(data_y[20:25], dtype=torch.float32)
    torch.save(valid_x, './.data/NCLT/valid/state.pt')
    torch.save(valid_y, './.data/NCLT/valid/obs.pt')
    print(valid_x.shape)
    print(valid_y.shape)


###### Test data ######
if mode == 'test':
    # data_x = sio.loadmat('data_x.mat')['data_x']
    # data_y = sio.loadmat('data_y.mat')['data_y']


    # test_x = torch.tensor(data_x[17:, [0,2,4,1,3,5]], dtype=torch.float32)
    # test_y = torch.tensor(data_y[17:], dtype=torch.float32)
    # torch.save(test_x, './.data/NCLT/test/state.pt')
    # torch.save(test_y, './.data/NCLT/test/obs.pt')
    # print(test_x.shape)
    # print(test_y.shape)

    ### Long Test Data

    data_x = sio.loadmat('data_x.mat')['data_x']
    data_y = sio.loadmat('data_y.mat')['data_y']

    test_x = torch.tensor(data_x[:1, [0,2,4,1,3,5]], dtype=torch.float32)
    test_y = torch.tensor(data_y[:1], dtype=torch.float32)
    torch.save(test_x, './.data/NCLT/test/state.pt')
    torch.save(test_y, './.data/NCLT/test/obs.pt')
    print(test_x.shape)
    print(test_y.shape)
