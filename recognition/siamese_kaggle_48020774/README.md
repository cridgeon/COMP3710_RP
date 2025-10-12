<h1>Siamese Network classification of the ISIC 2020 Kaggle Challenge data set</h1>

<h2> Prerequisites </h2>

Ensure you have [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installed.

<h2> Notes </h2>

If not otherwise specified it is assumed that the working directory that commands are 
executed in is the `recognition/siamese_kaggle_48020774` directory.

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
- run the conda_setup script

  `./utility/conda_setup.sh`