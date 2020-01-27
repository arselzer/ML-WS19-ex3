#!/usr/bin/env python

import concurrent.futures
import pandas as pd
import urllib
import pathlib
from threading import Thread
import subprocess

def crop_image(url, name, imageid, w, h, x_off, y_off):
	path = "persons/" + name + "/" + str(imageid) + ".jpg"
	outPath = "persons-cropped/" + name + "/" + str(imageid) + ".jpg"

	if pathlib.Path(path).exists():
		try:
			process = subprocess.Popen(['convert', path, '-crop', str(w) + "x" + str(h) + "+" + str(x_off) + "+" + str(y_off), outPath])
			out, err = process.communicate()
			print("cropped image " + path + " to " + outPath)
		except Exception as e:
			print(e)
		
people = pd.read_csv("dev_people.txt")

for i, row in people.iterrows():
	#print(row["person"])
	name = row["person"]
	pathlib.Path("persons-cropped/" + name).mkdir(parents=True, exist_ok=True)
pathlib.Path("persons-cropped").mkdir(parents=True, exist_ok=True)

threadPool = concurrent.futures.ThreadPoolExecutor(max_workers=12)

images = pd.read_csv("dev_urls.txt", sep="\t")
for i, row in images.iterrows():
	person = row["person"]
	imagenum = row["imagenum"]
	url = row["url"]
	rect = row["rect"]
	coords = rect.split(",")
	x0 = int(coords[0])
	y0 = int(coords[1])
	x1 = int(coords[2])
	y1 = int(coords[3])
		
	w = x1 - x0
	h = y1 - y0

	threadPool.submit(crop_image, url, person, imagenum, w, h, x0, y0)
	# fetch_image(url, person, imagenum)

#print(data)
