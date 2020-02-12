# Task 3.1.5

We worked with python 3 and tensorflow 2.1, and also with foolbox 2.

to install dependencies run
`python3 -m pip install -r requirements.txt`

## Dataset

The file `eval_people.txt` contains a list of all people (labels) and the file `eval_urls.txt` contains a list of images, URLs and the area to crop out

## Data preprocessing, training and adversarial generation

1) fetch the images (less than 20% are available and ok in the end)
`python3 fetch_images.py`

2) scale the images
`python3 crop_images.py`

3) perform the train test split
 `python3 split_data.py && ./copy_missing_images.sh`

4) Train the model: run jupyter notebook and the file `VGGface_Final.ipynb`

5) to predict an image run
`python3 predict_image.py image_path` 

6) to run adversarial image generation see `tensorflow_adversarial_attack.ipynb`
