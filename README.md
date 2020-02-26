# Task 3.1.5

## Installing dependencies 

We worked with python 3 and tensorflow 2.1, and also with foolbox 2.

First, you need to manually install `python3`, `pip` and `imagemagick` via your system's package manager.
This needs to be done manually, as each distribution has different package names/versions and commands to install them.

To install other dependencies run
`python3 -m pip install -r requirements.txt`

## Dataset

The file `eval_people.txt` contains a list of all people (labels) and the file `eval_urls.txt` contains a list of images, URLs and the area to crop out

## Data preprocessing, training and adversarial generation

1) fetch the images (less than 20% are available and ok in the end)
`python3 fetch_images.py`

2) scale the images
`python3 crop_images.py`

3) perform the train test split
`python3 split_data.py`

4) to train the model from scratch with the generated training and test data, run
`python3 VGGFace_Final.py`
   In the case where a saved model needs to be trained further, for example with attack data, one can run the command with the following parameters:
`python3 VGGFace_Final.py test_data_path validation_data_path model_path`
   If a model needs to be trained from scratch on non-standard data, the `model_path` argument can be omitted.

5) to predict an image run
`python3 predict_image.py model_path image_path` 
Example: ./predict\_image.py model.h5 persons-cropped-test/Adriana\ Lima/115.jpg

6) to run adversarial image generation see `tensorflow_adversarial_attack.ipynb`

7) to generate statistics run `generate_statistics.ipynb` interactively or

`./generate_statistics.py model method`, method is either 'gaussian' or 'saliency'

Example: `./generate_statistics.py model.h5 gaussian`
