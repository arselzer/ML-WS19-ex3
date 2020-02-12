#!/bin/bash
# terrible shell script that copies images from the training set to the test set where there are no images in the test set

while IFS= read -r dir; do file="$(ls "persons-cropped/$dir" | head -n 1)"; mkdir "persons-cropped-test/${dir}" && mv "persons-cropped/${dir}/${file}" "persons-cropped-test/${dir}/${file}"; done <<< $(diff persons-cropped persons-cropped-test | grep "Only in" | cut -d':' -f2 | cut -b2-)
