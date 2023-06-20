# Starting point for E2MIP challenge

You can use the code from this repository as starting point for your work on the [E2MIP challenge](https://e2mip.github.io/) on the LIDC-IDRI dataset.
It provides code for preprocessing and segmentation of 3D data.

To install the requirements for this sample code `$ pip install -r requirements.txt`, the code is tested with python version 3.10.9.


### Run Classification

#### Data path:
Set the following parameters to the paths of the folders, created by this [repository](https://github.com/XRad-Ulm/E2MIP_LIDCI-IDRI_data).
It provides code to create folders with the datasets in the same way as the data folders that will be used to train and test your submitted code.
* **--training_data_path**: str, e.g. "=training_data"
* **--testing_data_path**: str, e.g. "=testing_data_segmentation"
* **--testing_data_solution_path**: str, e.g. "=testing_data_solution_segmentation"

#### Data Preprocessing
In the sample code you can specify the data preprocessing with the following arguments:
* --batch_size: int, default=8
* --patch_size: tuple, default=(64,64,64)

Preprocessed data will be saved in a folder named "my_training_data". 
First time running the code will take some time, after that, preprocessed data will be loaded from that folder directly.

#### Train:
The sample code uses a simple 3D U-Net model, that can be used as starting point for the challenge.
In the sample code you can specify the training algorithm with the following arguments
* **--train=True**: bool, default=False
* --epochs: int, default=100
* --lr: float, default=0.01

#### Test:
In the sample code you can specify the testing algorithm with the following arguments:
* **--test=True**: bool, default=False
* **--model_path=[path_to_model.pth]**: str (If --train=True, this argument is being ignored and the newly trained model is being tested)

When required parameters specified, the predicted volumes are being saved in a "testing_data_prediction" folder.

#### Evaluating Dice Score:
To calculate the dice score on your test dataset, specify the following required argument:
* **--testing_data_solution_path=[path_to_testing_data_solution folder]** 

For further questions about this code, please contact luisa.gallee@uni-ulm.de
