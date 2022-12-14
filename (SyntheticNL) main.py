from GSSFiltering.model import SyntheticNLModel

TRAIN = True
# TRAIN = False
is_mismatch = False

if TRAIN:
    data_model = SyntheticNLModel(is_mismatch=is_mismatch, is_train=True).generate_data()
data_model = SyntheticNLModel(is_mismatch=is_mismatch, is_train=False).generate_data()



from GSSFiltering.model import SyntheticNLModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester


train_iter = 500
batch_size = 1
alter_num = 1

is_mismatch = False

# S_KalmanNet
test_list = ['best']

loss_list_Kalman = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

if TRAIN:
    # KalmanNet
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(
            SyntheticNLModel(is_train=True, is_mismatch=is_mismatch)), 
        data_path='./.data/syntheticNL/train/(true)', 
        save_path='(syntheticNL) KalmanNet.pt',
        mode=0)
    trainer_kalman.batch_size = batch_size
    trainer_kalman.alter_num = alter_num

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(
            SyntheticNLModel(is_train=True, is_mismatch=is_mismatch)), 
        data_path='./.data/syntheticNL/train/(true)', 
        save_path='(syntheticNL, v2) KalmanNet.pt',
        mode=0)
    trainer_kalman_v2.batch_size = batch_size
    trainer_kalman_v2.alter_num = alter_num    

    # S_KalmanNet 
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(
            SyntheticNLModel(is_train=True, is_mismatch=is_mismatch)), 
        data_path='./.data/syntheticNL/train/(true)', 
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
                            SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                        data_path = './.data/syntheticNL/test/(true)',
                        model_path = './.model_saved/(syntheticNL) Split_KalmanNet_' + str(trainer_split.train_count) + '.pt',
                        is_validation=True
                        )    
            )       
        
        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)    
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                        filter = KalmanNet_Filter(
                            SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                        data_path = './.data/syntheticNL/test/(true)',
                        model_path = './.model_saved/(syntheticNL) KalmanNet_' + str(trainer_kalman.train_count) + '.pt',
                        is_validation=True
                        )    
            )

        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                        filter = KalmanNet_Filter_v2(
                            SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                        data_path = './.data/syntheticNL/test/(true)',
                        model_path = './.model_saved/(syntheticNL, v2) KalmanNet_' + str(trainer_kalman_v2.train_count) + '.pt',
                        is_validation=True
                        )    
            )    

 
tester_ekf = Tester(
            filter = Extended_Kalman_Filter(
                SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
            data_path = './.data/syntheticNL/test/(true)',
            model_path = 'EKF'
            )   
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:

    tester_kf = Tester(
                filter = KalmanNet_Filter(
                    SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                data_path = './.data/syntheticNL/test/(true)',
                model_path = './.model_saved/(syntheticNL) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf2 = Tester(
                filter = KalmanNet_Filter_v2(
                    SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                data_path = './.data/syntheticNL/test/(true)',
                model_path = './.model_saved/(syntheticNL, v2) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]    

    tester_skf = Tester(
                filter = Split_KalmanNet_Filter(
                    SyntheticNLModel(is_train=False, is_mismatch=is_mismatch)), 
                data_path = './.data/syntheticNL/test/(true)',
                model_path = './.model_saved/(syntheticNL) Split_KalmanNet_' + elem + '.pt'
                )
    loss_list_Split += [tester_skf.loss.item()]

print(loss_ekf)
print(loss_list_Kalman)
print(loss_list_Kalman_v2)
print(loss_list_Split)