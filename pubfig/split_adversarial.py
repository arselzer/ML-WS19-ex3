## special script for special purpose
# only to seperate images with odd numbers from adversarial folder

import os
import random

directory = "./big/persons-adversarial"
destination_directory = "./big/persons-adversiarial-odds"

if not (os.path.isdir(os.path.join(destination_directory))):
    os.mkdir(os.path.join(destination_directory))

images_sum = 0
images_dropped = 0
for filename in os.listdir(directory):
    if not (os.path.isdir(os.path.join(destination_directory, filename))):
        os.mkdir(os.path.join(destination_directory, filename))
    for image in os.listdir(directory + "/" + filename):
        images_sum += 1
        if (int(image.split(".")[0]) % 2 == 0):
            images_dropped += 1
            os.rename(os.path.join(directory, filename, image), os.path.join(destination_directory, filename, image))
            print(os.path.join(directory, filename, image))

print("Alle: ", images_sum)
print("Moved: ", images_dropped)
print("Moved-Quote: ", round(images_dropped/images_sum, 4))
print("Stays-Quote: ", round(1-(images_dropped/images_sum), 4))

