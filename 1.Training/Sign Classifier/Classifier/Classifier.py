from tscLib import *

ensur_dir(model_fldr)
ensur_dir(model_weights)

model = createModel(model_arch)


if load_weights == 1:
	loadWeights(model,weights_dir)
	if continue_training == 1:
		steps = countEpoch(name_train)
		stepsval = countEpoch(name_test)
		datagen = data_generator()
		valgen = val_generator()
		SC = trainBatch(model,datagen,steps,valgen,stepsval)

else:
	steps = countEpoch(name_train)
	stepsval = countEpoch(name_test)
	datagen = data_generator()
	valgen = val_generator()
	SC = trainBatch(model,datagen,steps,valgen,stepsval)

logModel(SC)
