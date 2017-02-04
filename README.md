# Behavioral Cloning for Self-Driving Cars

The goal of this project is to train a deep neural network to clone driving behavior by using images to predict steering angles. Once the model is trained, it will be used to drive a car autonomously around a test track in Udacity's driving simulator.


## Requirements and Installation

1. Install Anaconda or Miniconda with Python 3.5 and Udacity's carnd-term1 conda environment
    * Follow Udacity's [guide for getting your environment setup](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)
2. Clone this repository
    * `git clone https://github.com/ncondo/CarND-Behavioral-Cloning`
3. Download Udacity's sample data and save it in the same directory as model.py
    * [Dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
4. Download Udacity's driving simulator
    * [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
    * [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
    * [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
    * [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)
    
## Files and Usage

1. model.py
    * Contains all code for reading in training data and training a model to predict steering angles.
    * `python model.py' will train a new model and save the output architecture as model.json and weights as model.h5.
    * Running model.py will overwrite current model.json and model.h5 files - to prevent this rename models you want to save.
2. model.json
    * Model architecture saved after running `python model.py`
3. model.h5
    * Model weights saved after running `python model.py`
4. drive.py
    * Contains code to serve predictions from a trained model to Udacity's simulator for autonomous driving.
    * `python drive.py model.json` will run the model (must have simulator open and in autonomous mode).
    * `python drive.py model.json run1` will run the model and save all images from the run in a new directory with name run1
5. video.py
    * Contains code to create an mp4 video from the previously saved images from an autonomous run.
    * `python video.py run1`