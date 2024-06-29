
# Animal Classification 

This project is for pet management and care   department, supporting animal welfare,Support animal shelters in managing and categorizing animals effectively.

## Aims and objectives:

The broader aims of the project are:

⦁ Provide tools for pet owners, shelters, and veterinarians to easily identify and differentiate between cats and dogs, aiding in pet management and care.

⦁Assist in tracking and reuniting lost pets with their owners through accurate identification systems.

⦁Support animal shelters in managing and categorizing animals effectively.

## Objectives:
The primary objectives of this project are:

⦁Create a machine learning model capable of distinguishing between cat and dog images with high accuracy.

⦁Evaluate the performance of different algorithms (e.g.Convolutional Neural Networks) to determine the most effective approach.

⦁Extend the classification model to locate and identify cats and dogs within an image using object detection techniques.

## Abstract:
•A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

⦁One such technique is to use YOLOv5 with Roboflow model,which generates a small size trained model and makes ML integration easier.

⦁Collect and annotate a large dataset of cat and dog images,ensuring diversity in terms of breeds, backgrounds and lighting conditions.

⦁Split the dataset into training, validation, and test sets for robust model evaluation.

## Introduction

⦁This project aims to develop an advanced image classification and object detection system that can accurately identify and differentiate between cats and dogs in images. 

⦁Using machine learning and computer vision techniques, this system will serve as a foundational model with broad applications in various fields such as pet care, surveillance, and more.

⦁Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for this project as well.

⦁The practical applications of cat and dog image classification and object detection are diverse.

⦁These techniques can be used in smart home systems for pet monitoring, automated pet feeders and entertainment systems. 

⦁In the realm of animal welfare, accurate pet identification systems help in reuniting lost pets with their owners and managing animals in shelters 

## Literature Review

This project leverages YOLOv5 for both classification and detection, and it is trained and fine-tuned to achieve high accuracy and real-time performance. The developed model will be optimized for deployment across multiple platforms, including edge devices like the NVIDIA Jetson Nano, making it accessible for practical use cases such as pet identification, smart home surveillance, and animal welfare management.

## Proposed System
1)Study basics of machine learning and image recognition.

2)Start with implementation
```bash
 ➢ Front-end development
 ➢ Back-end developmen
```
3)Testing, analysing and improvising the model. An application using python and Roboflow and  machine learning algorithms to identify the image of cat or dog

4)Use datasets to classify whether the animal in the image is a cat or dog

## Methodology

Gather a diverse set of cat and dog images from public datasets and other sources.

Annotate the images with labels for classification and bounding boxes for object detection.

Train YOLOv5 on the prepared dataset.

Evaluate the model using metrics such as accuracy, precision, recall, and mean Average Precision (mAP).

Experiment with different data augmentation techniques and hyperparameter settings.

Apply transfer learning to improve model performance.

Develop a user-friendly application for real-time classification and detection.

Test the application in real-world scenarios to ensure reliability and efficiency.

## Installation
Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
Create Swap file

```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```
Cuda env in bashrc

```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
Update & Upgrade

```bash
sudo apt-get update
sudo apt-get upgrade
```
Install some required Packages
```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow
```
Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
Install Torchvision
```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
Clone Yolov5
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
Download weights and Test Yolov5 Installation on webcam
```bash
sudo python detect.py
sudo python detect.py --weights yolov5s.pt --source 0

```
## Dataset Training

We used Google Colab And Roboflow
train your model on colab and download the weights and past them into yolov5 folder link of project

colab file given in repo

## Running Animal Detection Model

source 0 -----> webcam
```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
## Advantages
1)classify and detect cats and dogs in various images, facilitating applications in pet care, surveillance, and animal welfare.

2)It helps in advancements in Veterinary Care

3)Contributions to Animal Welfare

4)Enhanced Pet Identification and Management

## Applications

1)It can be used in Smart Home Systems where   Automated pet monitoring is required.

2)Driving innovation in pet technology products, such as automated feeders and entertainment systems that recognize and respond to specific pets.

3)Enhanced Pet Identification and Management  
Lost Pet Recovery: Accurate classification and detection can assist in identifying lost pets and reuniting them with their owners through automated systems in shelters and communities.

## Future Scope

Expand Dataset: 
Include more diverse images and additional animal categories.

Enhance Model: 
Experiment with more advanced architectures and fine-tuning techniques.

Application Development:
Create more sophisticated applications for various use cases, such as veterinary diagnostics and animal behavior analysis.

## Conclusion

The project successfully developed a high-accuracy, real-time image classification for cats and dogs using YOLOv5. This system has potential applications in pet care, smart home systems, and animal welfare management. Additionally, it contributes to the advancement of computer vision techniques and provides a foundation for future research and development.



## Refrences:

1]Roboflow :- https://roboflow.com/

2] Datasets or images used: kaggle.com
