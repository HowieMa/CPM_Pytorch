## CMU Hand dataset

This is the sample of [CMU Hand dataset](http://domedb.perception.cs.cmu.edu/handdb.html)    


### Get dataset 

You can run 

` wget http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip `


to get dataset of **Hands with Manual Keypoint Annotations**  


### Dataset organization
The raw data is organized in the following way:  



### Preprocess
To make the dataset more appliable, I make some changes to its organization.  
You can simply run

`sh preprocess.sh `

or you can run the following code by yourself. 

````
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

````

to make it.   

Thus dataset is orgranized in the following way:  



 



