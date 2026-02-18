# Bone-Fracture-Detection (G12)

Welcome to my GitHub repo for my Bone Fracture Detection project in Musculoskeletal Radiographs (X-rays).

# Introduction

Bone fractures are common injuries generally diagnosed through X-rays. Deep learning models for fracture detection can aid clinicians in timely and accurate diagnoses of fractures. However, these models typically focus on specific regions of the body, limiting their generalizability. To address this, we trained Faster R-CNN,  RetinaNet, and YOLO12n object detection models on the FracAtlas dataset, which consists of X-rays of several regions of the body.

In this repo, look in the **models** folder for three sub-folders, namely **FasterRCNN**, **RetinaNet**, and **YOLOv12**. Each of these three sub-folders contains relevant code and results for their respective models. Moreover, in each of these sub-folders, you will find a README. Please refer to the README (one for each model) in order to understand which packages (and their versions) are needed for model execution. Moreover, the README contains intructions for how to run both the model, and related scripts for analyses and visualizations, on your local machine.

# Dataset Information

Dataset Paper:

* I. Abedeen, M. Rahman, F. Prottyasha, T. Ahmed, T. Chowdhury and S. Shatabda, "FracAtlas: A Dataset for Fracture Classification, Localisation and Segmentation of Musculoskeletal Radiographs," Scientific Data, vol. 10, p. 521, 2023. https://www.nature.com/articles/s41597-023-02432-4.

Link to dataset:

* https://figshare.com/articles/dataset/The\_dataset/22363012
