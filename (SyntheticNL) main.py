from GSSFiltering.model import SyntheticNLModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import numpy as np

TRAIN = True
# TRAIN = False

if TRAIN:
    SyntheticNLModel(mode='train').generate_data()
    SyntheticNLModel(mode='valid').generate_data()
SyntheticNLModel(mode='test').generate_data()

train_iter = 500
batch_size = 1
alter_num = 1

# S_KalmanNet
test_list = ['500']

loss_list_Kalman = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

valid_loss_Kalman = []
valid_loss_Kalman_v2 = []
valid_loss_Split = []

if TRAIN:
    # KalmanNet
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(
            SyntheticNLModel(mode='train')), 
        data_path='./.data/syntheticNL/train/', 
        save_path='(syntheticNL) KalmanNet.pt',
        mode=0)
    trainer_kalman.batch_size = batch_size
    trainer_kalman.alter_num = alter_num

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(
            SyntheticNLModel(mode='train')), 
        data_path='./.data/syntheticNL/train/', 
        save_path='(syntheticNL, v2) KalmanNet.pt',
        mode=0)
    trainer_kalman_v2.batch_size = batch_size
    trainer_kalman_v2.alter_num = alter_num    

    # S_KalmanNet 
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(
            SyntheticNLModel(mode='train')), 
        data_path='./.data/syntheticNL/train/', 
        save_path='(syntheticNL) Split_KalmanNet.pt',
        mode=1)
    trainer_split.batch_size = batch_size
    trainer_split.alter_num = alter_num    

    
    for i in range(train_iter):

        trainer_split.train_batch()
        trainer_split.dnn.reset(clean_history=True)      
        if trainer_split.train_count % trainer_split.save_num == 0:
            trainer_split.validate(
                Tester(
                        filter = Split_KalmanNet_Filter(
                            SyntheticNLModel(mode='valid')), 
                        data_path = './.data/syntheticNL/valid/',
                        model_path = './.model_saved/(syntheticNL) Split_KalmanNet_' + str(trainer_split.train_count) + '.pt',
                        is_validation=True
                        )    
            )
            valid_loss_Split += [trainer_split.valid_loss]   
            

    
        
        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)    
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                        filter = KalmanNet_Filter(
                            SyntheticNLModel(mode='valid')), 
                        data_path = './.data/syntheticNL/valid/',
                        model_path = './.model_saved/(syntheticNL) KalmanNet_' + str(trainer_kalman.train_count) + '.pt',
                        is_validation=True
                        )    
            )
            valid_loss_Kalman += [trainer_kalman.valid_loss]

        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                        filter = KalmanNet_Filter_v2(
                            SyntheticNLModel(mode='valid')), 
                        data_path = './.data/syntheticNL/valid/',
                        model_path = './.model_saved/(syntheticNL, v2) KalmanNet_' + str(trainer_kalman_v2.train_count) + '.pt',
                        is_validation=True
                        )    
            )    
            valid_loss_Kalman_v2 += [trainer_kalman_v2.valid_loss]



    validator_ekf = Tester(
                filter = Extended_Kalman_Filter(
                    SyntheticNLModel(mode='valid')), 
                data_path = './.data/syntheticNL/valid/',
                model_path = 'EKF'
                )   
    loss_ekf = [validator_ekf.loss.item()]

    np.save('valid_loss_ekf.npy', np.array(loss_ekf))
    np.save('valid_loss_kalman.npy', np.array(valid_loss_Kalman))
    np.save('valid_loss_kalman_v2.npy', np.array(valid_loss_Kalman_v2))
    np.save('valid_loss_split.npy', np.array(valid_loss_Split))

 
tester_ekf = Tester(
            filter = Extended_Kalman_Filter(
                SyntheticNLModel(mode='test')), 
            data_path = './.data/syntheticNL/test/',
            model_path = 'EKF'
            )   
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:

    tester_kf = Tester(
                filter = KalmanNet_Filter(
                    SyntheticNLModel(mode='test')), 
                data_path = './.data/syntheticNL/test/',
                model_path = './.model_saved/(syntheticNL) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf2 = Tester(
                filter = KalmanNet_Filter_v2(
                    SyntheticNLModel(mode='test')), 
                data_path = './.data/syntheticNL/test/',
                model_path = './.model_saved/(syntheticNL, v2) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]    

    tester_skf = Tester(
                filter = Split_KalmanNet_Filter(
                    SyntheticNLModel(mode='test')), 
                data_path = './.data/syntheticNL/test/',
                model_path = './.model_saved/(syntheticNL) Split_KalmanNet_' + elem + '.pt'
                )
    loss_list_Split += [tester_skf.loss.item()]

print(loss_ekf)
print(loss_list_Kalman)
print(loss_list_Kalman_v2)
print(loss_list_Split)