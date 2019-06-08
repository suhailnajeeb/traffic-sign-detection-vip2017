# Training challenge type classifier 
-------------------------------------


# GENERATING FEATURES FROM FRAMES
-------------------------------------
- Features are extracted by executing the extractFeatures.py script :
			
			$ python extractFeatures.py

- We extracted frames from the videos at the very begining, and all our training uses extracted frames instead of direct from video because our computers intially had trouble with openCV 's VideoCapture() function

- Using extractFeatures.py we generate npz files to be used for training


- Directory parameters in the extractFeatures.py script must be edited to the locations of the frames:
		
	Location of Synthesized Frames:
		loc = '/media/user/drive/Frames/'	



- For getting features from training dataset, set video selection parameters as follows:

		N_vid = [0,0]		# video sequence
		effects = [0,12]	# challenge type 1-12 (1-11 for syn automatically adjusted )
		levels =[1,5]		# challenge level 1-5
		syn = 1			# make 1 to include Synthesized videos
		only_syn = 0		# make 1 and syn = 1 to include only Synthesized videos			


- For getting features from test dataset, set video selection parameters as follows:

		N_vid = [0,1]		# video sequence
		effects = [0,12]	# challenge type 1-12 (1-11 for syn automatically adjusted )
		levels =[1,5]		# challenge level 1-5
		syn = 1			# make 1 to include Synthesized videos
		only_syn = 0		# make 1 and syn = 1 to include only Synthesized videos	


- We extracted features in two stages; 
		
		Feature Set B_train_125_125.npz:
		----------------------------------------------------------------------
			contains all trainset videos, 			N_vid = [0,0]
			all challenges including no challenge 		effects = [0,12]
			and challenge levels 2 and 3			levels = [2,3]

		Feature Set B_test_125_125.npz:
		----------------------------------------------------------------------
			contains all testset videos, 			N_vid = [0,1]
			all challenges including no challenge 		effects = [0,12]
			and challenge levels 2 and 3			levels = [2,3]

		Feature Set C_train_125_125.npz:
		----------------------------------------------------------------------			
			contains all trainset videos, 			N_vid = [0,0]
			all challenges *excluding* no challenge 	effects = [1,12]
			and challenge levels 4 and 5			levels = [4,5]

		Feature Set C_test_125_125.npz:
		----------------------------------------------------------------------			
			contains all testset videos, 			N_vid = [0,1]
			all challenges *excluding* no challenge 	effects = [1,12]
			and challenge levels 4 and 5			levels = [4,5]


# TRAINING MODEL
------------------
- Model is trained using trainRCNN.py script

			$ python trainRCNN.py

- Parameters for load/train model must be set in the scripty file
		
		load_model = 0				# setting to 1 will load model with name given in load_model_name
		continue_training = 0			# setting to 1 will continue training loaded model with train_features
		evaluate_on_test_data = 0		# setting to 1 will evaluate model on test_features

- The feature names must be include in train_feat_name and test_feat_name, and the model must be given a name (model_name)




