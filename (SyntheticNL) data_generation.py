from GSSFiltering.model import SyntheticNLModel

is_mismatch = False

data_model = SyntheticNLModel(is_mismatch=is_mismatch, is_train=True).generate_data()
data_model = SyntheticNLModel(is_mismatch=is_mismatch, is_train=False).generate_data()
