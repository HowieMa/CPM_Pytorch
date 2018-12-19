# get data
wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip
unzip hand_labels.zip

cd hand_labels/manual_train/
mkdir data
mkdir label
mv *jpg data/
mv *json label/

cd ../manual_test/
mkdir data
mkdir label
mv *jpg data/
mv *json label/
cd ..

mv manual_train train
mv manual_test test

python crop_hand.py


