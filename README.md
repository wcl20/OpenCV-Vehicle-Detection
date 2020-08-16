# OpenCV Vehicle Detection
Vehicle detection using HOG features and SVM.

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```

### dataset.py
Convert images into Numpy arrays and save in HDF5 file. The data.h5 file contains:
  * an array of vehicle images
  * an array of background images
  
Image dataset: [Link](https://www.gti.ssr.upm.es/data/Vehicle_database.html)

### feature.py
Convert images into a vector of features using HOG.

### model.py
Trains a SVM model to classify vehicles from background images. The model is saved in model.pickle file.

### slider.py
The Sliding Window approach slides a window along the region of interest to detect vehicles in the frame.

### heatmap.py
Used as Non-Maximum suppression technique to find bounding boxes and remove false positives.

### main.py
The pipeline of vehicle detection. Outputs a video with detected vehicles.
