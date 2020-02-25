#!/usr/bin/env python

import concurrent.futures
import pandas as pd
import urllib.request
import pathlib
import hashlib
from threading import Thread

def fetch_image(url, name, imageid, correct_md5sum):
	path = "persons/" + name + "/" + str(imageid) + ".jpg"
	#urllib.request.urlretrieve(url, path) 
	try:
		response = urllib.request.urlopen(url, timeout=12)
		img = response.read()
		image_md5sum = hashlib.md5(img).hexdigest()

		if image_md5sum == correct_md5sum:
			file = open(path, "wb+")
			file.write(img)
			print("fetched " + url + " to " + path)
		else:
			print("hashes do not match up (image " + imageid + " of " + name + "): " + image_md5sum + " vs actual: " + correct_md5sum)
	except Exception as e:
		print("timeout fetching " + url + e)

print("Fetching images...")

people = pd.read_csv("eval_people.txt")

pathlib.Path("persons").mkdir(parents=True, exist_ok=True)

for i, row in people.iterrows():
	#print(row["person"])
	name = row["person"]
	pathlib.Path("persons/" + name).mkdir(parents=True, exist_ok=True)

threadPool = concurrent.futures.ThreadPoolExecutor(max_workers=100)

data = pd.read_csv("eval_urls.txt", sep="\t")
for i, row in data.iterrows():
	person = row["person"]
	imagenum = row["imagenum"]
	url = row["url"]
	rect = row["rect"]
	md5sum = row["md5sum"]
	threadPool.submit(fetch_image, url, person, imagenum, md5sum)
	# fetch_image(url, person, imagenum)

#print(data)
