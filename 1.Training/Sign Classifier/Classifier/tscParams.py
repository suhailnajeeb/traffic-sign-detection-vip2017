# ----------------------------- TSC Settings -----------------------------------

# name of model architecture to use as defined in bcModels
model_arch = "sc_v6"


name_train = ['Model1_00','Model1_00','Model1_02',....,''] #name of the npz files
name_test =['ZoomNC_test_64_64','ZoomCh1_test_64_64',...''] 	#explicit name of the validation npzs


# Model Description
model_name = "Model1"
load_model = "Model1"                          # name of model to load to test/continue training

desc = "Trained with ZNZMAN_NC Data "

# Set to 1 to validate on test set;

test_set = 1

# Set to 1 to load weights instead of training
load_weights = 0

# Set to 1 to continue training model with loaded weights
continue_training = 0

# CNN Parameters
batch_size = 	64
nb_epoch = 	10
nb_classes = 	24

img_rows = 64
img_cols = 64

model_dir = '/.../Models/'	#folder to store model
feat_dir = '/.../Features/'	#folder where features are stored

model_fldr = model_dir + model_name + "/"
model_weights = model_fldr + "weights/"
weights_dir = model_dir + load_model + "/best.hdf5"
