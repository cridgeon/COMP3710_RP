<h1>Siamese Network classification of the ISIC 2020 Kaggle Challenge data set</h1>

<h2> Prerequisites </h2>

Ensure you have [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed.

<h2> Notes </h2>

If not otherwise specified it is assumed that the working directory that commands are 
executed in is the `recognition/siamese_kaggle_48020774` directory.

<h2> Model and Training Breakdown </h2>

<h3> Data </h3>

The main concern while using the dataset was the large disparity between the number of 
positive (malignant) and negative (benign) classifications. For the entire dataset, 
only 0.018% of the images have a malignant classification. This large disparity would be 
certain to cause issues during training, as there is a heavy bias towards a model that 
classifies all images as benign. Thus a hyperparameter defining the desired split of 
positive and negative classfications was added and con be configured in 
`utility/config.json`, by changing the value of the `class_split` parameter.

Note that this parameter is the main one used for determining the number of images loaded
into memory. Since there are more negative classifications, a lower percentage will allow
for more of the data to be used for training, while a higher percentage means that less
data can be used in order to maintain the correct ratio.

<h3> hyperparameters </h3>

Below is a list of hyperparameters and their functions

| Name | Function |
| ---- | -------- |
| `data_dir`         | Directory where the dataset is stored relative to working directory|
| `train_split`      | Fraction of data used for training (rest for validation/testing).  |
| `class_split`      | Proportion of positive (malignant) samples in the dataset.   |
| `max_images   `    | The maximum number of images loaded into mem. To prevent mem errors|
| `anchor_count`     | The number of images to reserve for use as anchors.    |
| `image_width`      | Width to resize images to during preprocessing.        |
| `image_height`     | Height to resize images to during preprocessing.       |
| `batch_size`       | Number of samples per training batch.                  |
| `epochs`           | Number of training epochs.                             |
| `learning_rate`    | Learning rate for the optimizer.                       |
| `model_save_path`  | File path to save the trained model.                   |

<h2> Using the model </h2>

<h3> Download the dataset </h3>

**Automatic Download**
- Run the data_download script
  
  `./utility/data_download.sh`

**Manual Download**
- Navigate to the [ISIC Kaggle Dataset Page](https://challenge2020.isic-archive.com/)  
- Download the training jpeg zip file
- Download the training metadata
- Run `mkdir data`
- Unzip the downloaded file into the `data` folder
- move the training metadata into the `data` folder
- run `mkdir data/images && mv data/train/* data/images && rmdir data/train`
- The file structure should now be as follows:
  ```
  recognition/siamese_kaggle_48020774
    - data
      - images  (contains the 33,126 training images)
      - ISIC_2020_Training_GroundTruth.csv
    - utility
    .
    .
    .
  ```

<h3> Setting up the conda environment </h3>
Run the conda_setup script:

```    
./utility/conda_setup.sh
```

Activate the environment:

```
conda activate siamese-48020774
```