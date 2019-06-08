import numpy as np
import os

#directories
feat_dir = '/../Features/'		#directory for source features
save_dir = feat_dir

#feature names which need to be shuffled and split
feats= ['ZoomCh1','NoZoomCh1','ManCh1',.....'','',''] 
save_name = 'Model1' #feature save name

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

feat_path = [feat_dir + feat + '_train_64_64.npz' for feat in feats]
parts = len(feat_path)

X = []
Y = []

print "\nInitializing"

# loading feats
for path in feat_path:
    npzfile = np.load(path)
    X.extend(npzfile['X_train'])
    Y.extend(npzfile['Y_train'])
    print "Loading: %s" % path

print "\nLoaded: %d entries" % len(X)

# shuffling
print "\nShuffling . . ."
X = np.array(X)
Y = np.array(Y)

idx = np.random.permutation(len(X))
X = X[idx]
Y = Y[idx]

n_entries = len(X)/parts
start_entry = 0
end_entry = start_entry + n_entries

# splitting
total_entries = 0
print "\nSpliting . . ."

for idx2 in range(parts-1):
    x = X[start_entry : end_entry]
    y = Y[start_entry : end_entry]
    np.savez(save_dir + save_name + '_%02d'%idx2,X_train=x,Y_train=y)

    print "Saved: %d entries" % len(y)
    total_entries += len(y)

    start_entry = end_entry
    end_entry = start_entry + n_entries

# putting remaning entries into last part

x = X[start_entry : ]
y = Y[start_entry : ]

np.savez(save_dir + save_name + '_%02d'%(parts-1),X_train=x,Y_train=y)

print "Saved: %d entries" % len(y)
total_entries += len(y)

print "\nSplitting Complete. Generated %d files containing %d entries total\n" % (parts,total_entries)
