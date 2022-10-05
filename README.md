# Satellite Image Land Cover Segmentation using U-net 

This GitHub repository is developed by Srimannarayana Baratam and Georgios Apostolides as a part of Computer Vision by Deep Learning (CS4245) course offered at TU Delft. The implementation of the code was done using PyTorch, it uses U-net architecture to perform multi-class semantic segmentation.  The repository from which our implementation has been derived can be found [[here]](https://github.com/milesial/Pytorch-UNet). A well articulated blog is also available [[here]](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) for the project by the authors of this repository.

## Google Colab Wrapper
For testing the repository, a google colab wrapper is also provided which explains in detail how to execute the code along with insights. Just download the "colab_wrapper.ipynb" file from the repository and open in your colab. Instructions are available there to clone this repository directly to your drive and train using GPU runtime.

## Dataset
The dataset we used is taken from the DeepGlobe Challenge of Land Cover Segmentation in 2018. [[DeepGlobeChallenges]](http://deepglobe.org/challenge.html)  However, the server for the challenge is no longer available for submission and evaluation of solutions. [[DeepGlobe2018 Server]](https://competitions.codalab.org/competitions/18468) and the validation and test set are not accompanied by labels. For this reason we are using **only the training set**  of the challenge and we are further splitting it into validation and test set to be able to evaluate our solution.  The original dataset can be downloaded from Kaggle [[DeepGLobe2018 Original Dataset]](https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset) here and the dataset we use can be downloaded from [[Link]](https://www.kaggle.com/geoap96/deepglobe2018-landcover-segmentation-traindataset) separated into the training/validation and test set we used for our model.

## Files Explanation
In this section we will present the different files inside the repository as well as an explanation about their functionality


|File Name| Explanation / Function |
|---------|------------|
|`U-net_markdown.ipynb`<img width=90/>| Used as a wrapper to run the train and predict scripts.|
|`train.py` | Used for the training of the network.  |
|`predict.py`|Used to perform a prediction on the test dataset. |
|`diceloss.py` | It's a function used to calculate the dice loss.|
|`classcount.py`| Calculates the weights to be used for the weighted cross entropy by counting the pixel number of each class in the train dataset.|
|`distribution.py`| Used to evaluate the pixel numbers of the validation and training set and visualize them via  bar chart.|
|`dataset.py`| Used as a data loader during the training phase.|

## Training

The following flags can be used while training the model.

<ins>_Guidelines_<ins>

`-f` : Used to load a model already stored in memory. \
`-e` : Used to specify the Number of training epochs. \
`-l` : Used to specify the learning rate to be used for training. \
`-b` : Used to specify the batch size. \
`-v` : Used to specify the percentage of the validation split (1-100). \
`-s` : Used to specify the scale of the image to be used for training.
`-m` : Path to the folder where the masks are stored (e.g. `/data/masks_subset/`)
`-i` : Path to the folder where the images are stored (e.g. `/data/img_subset/`)

<ins>_Example:_<ins/>

Training the model for 100 epochs using 20% of the data as a validation split, learning rate is 4x10^-5, batch size is 2 and image scale is 20%

`!python3 train.py -e 100 -v 20.0 -l 4e-5 -b 2 -s 0.2`

## Prediction
<ins>_Guidelines_<ins>

`-m` : Used to specify the directory to the model. \
`-i` : Used to specify the directory of the images to test the model on. \
`-o` : Used to specify the directory in which the predictions will be outputted. \
`-s` : Used to specify the scale of the images to be used for predictions. \
`--viz:` Saves the predictions in the form of an image. \
(For best results used the same scale you used for training the model)

_Note:_ Inference of scale 0.2 takes approximately 10 minutes.

<ins>_Example_<ins>

Making a prediction on the full test set dataset using 30 epoch model trained on full data using a scale of 20%. The script  outputs the IoU score of the model.

```
%%time
!python predict.py -m data/checkpoints/model_ep30_full_data.pth -i data/<test_set_directory>/* -o predictions/ -s 0.2 --viz
```

## Lambda Lab SetUp

### setup Lambda Lab
1. ``ssh`` into lambda_lab server, e.g. `ssh team_051@138.2.47.80` (password is in https://discord.com/channels/984525101678612540/1020432788542980217)
2. clone the git repository using 
   ```
   git clone https://github.com/tayyabmujahid/land_cover_classification_unet.git
   ```
3. check out the branch of interest
   ```
   cd land_cover_classification_unet
   git checkout <branchname>
   ```
   
4. Build the docker image used for running the training
   ```
   docker build -t landcoverseg:latest -f train.Dockerfile .
   ```
   If multiple users are having different images, consider using a different tag instead of ``latest``
5. [optional] download the kaggle data set using
   ```
   ./download_kaggle_data.sh
   ```
   Note: one can use the -d option to specify a path to download to.
   
   
   

### run training in lambda labs
1. In case you changed the ``requirements.txt`` rebuild the container
2. Start your docker container using:
   ```
   docker run -it --ipc=host --gpus \"device=${CUDA_VISIBLE_DEVICES}\" -v /home/team_051/land_cover_classification_unet:/workspace -v /home/team_051/land_cover_classification_unet/data:/data -e WANDB_API_KEY=xxxx landcoverseg:latest bash
   ```
   Make sure you set the following correctly
   1. That the repo is mapped into the container, done by ``-v /home/team_051/land_cover_classification_unet:/workspace``
   1. In case you do not use the data from the repository to map them in by ``-v /home/team_051/land_cover_classification_unet/data:/data``
   1. Provide your `wandb` key to log your experiments ``-e WANDB_API_KEY=xxxx``. Optionally this can be set later within the container
2. Now since you are connected into the container you can run the training, e.g.
   ```
   python3 train.py -e 100 -v 20.0 -l 1e-5 -b 2 -s 0.2 -m "/data/masks_subset/" -i "/data/img_subset/"
   ```
   Where ``-m`` and `-i` define the path to the masks and images within the container. Most of the time this should point to the mapped data folder

__Note__: currently only the ``/workspace`` folder is mapped out from the container

