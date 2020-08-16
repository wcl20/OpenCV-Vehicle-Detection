import cv2
import glob
import h5py
import numpy as np

def get_data(filename):
    # Read HDF5 file
    hf = h5py.File(filename, "r")
    vehicles = hf.get("vehicles")[()]
    background = hf.get("background")[()]
    hf.close()
    return vehicles, background

def main(filename):
    # Load images
    print("Loading images ...")
    vehicles_files = glob.iglob("dataset/vehicles/*.png")
    background_files = glob.iglob("dataset/background/*.png")
    # Convert images to Numpy array
    vehicles = np.array([cv2.imread(file) for file in vehicles_files])
    background = np.array([cv2.imread(file) for file in background_files])

    print(f"Loaded {vehicles.shape[0]} vehicle images")
    print(f"Loaded {background.shape[0]} background images")

    # Save data in HDF5 file
    print(f"Saving data into {filename} ...")
    hf = h5py.File(filename, "w")
    hf.create_dataset("vehicles", data=vehicles)
    hf.create_dataset("background", data=background)
    hf.close()

    print("DONE!")

if __name__ == '__main__':
    main("data.h5")
