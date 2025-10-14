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
| `pn_sample_count`     | The number of images to reserve for use as samples to compare anchor image to.    |
| `batch_size`       | Number of samples per training batch.                  |
| `epochs`           | Number of training epochs.                             |
| `learning_rate`    | Learning rate for the optimizer.                       |
| `model_save_path`  | File path to save the trained model.                   |

<h3> Model Background </h3>

The Generic architecture of a Siamese Network features two identical subnetworks 
(twin networks) that share the same weights and parameters. Intended problem that
a Siamese Network solves is to determine whether two input samples are similar or 
different, classic examples being signature and face recognition. In this report
a Siamese Network is used to compare images of skin lesions with known malignant 
and benign melanoma samples, with the goal of classifying the unknown sample.

<h3> Model Architecture </h3>

The architecture used for the twin networks is a ResNet50 Convolutional Neural
Network. Which is proven be effective at extraction of key image features, while
maintaining relative minimality. The ResNet50 architecture features 49 
convolutional layers and a maxpooling layer. It is comprised of multiple blocks
that are that have a "bottleneck" design.

The bottleneck design is conprised of three convolutional layers. The first layer
is a 1x1 convolutional layer that reduces the number of channels, designed to
reduce dimensionality while maintaining key features. The second layer is a 3x3
convolutional layer with a stride of 2, used to extract spatial features. The 
final layer is another 1x1 convolutional layer that returns the dimensionality
to the original number of channels. 

![ResNet Bottleneck Block](resources/bottleneck.webp)

The full ResNet50 architecture employs 4 permutations of the bottleneck block,
arranged as shown below.

![ResNet50 Architecture](resources/resnet50.webp)

A more in depth explanation of the ResNet50 architecture can be found [here](https://blog.roboflow.com/what-is-resnet-50/).

Following the ResNet50 is an encoder head, which is a dense net that reduces the
output of the resnet, over a number of layers to a 256 dimensional vector. This 
vector represents the models encoding of an image into a kind of latent space.
This classifier, the resnet followed by encoder, is what is used as the twin 
network.

In making this a Siamese network, three of the classifiers are run at the same 
time, on an anchor image, known positive image and known negative image. Then,
a "distance" can be calculated between the latent encodings of the images. If
the distance to the positive image is smaller than the distance to the negative
image, the anchor image can be considered classified as positive and vice versa.

To account for this, after the triplet networks, a custom distance layer is added
that outputs the distances to from the anchor encoding to the others. Followed by
a loss layer, that calculates the Triplet loss, explained next, for the optimiser
to use. Finally, a display layer is added, which outputs a 1 if the distance to 
the positive image is greater than (or equal to) the distance to the negative.

<h3> Loss Function </h3>

The loss function used to evaluate the accuracy of the model is triplet loss.
>Triplet loss is a loss function where we compare a baseline (anchor) input to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to the negative (falsy) input is maximized. [[1]](https://builtin.com/machine-learning/siamese-network)
![triplet loss equation](./resources/triplet.webp)

The 'positive' input referred to should have the same classification as the anchor
image, while the negative one should not.

<h3> Training Process </h3>

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