from skimage import filters, segmentation, feature, future
from scipy import ndimage as ndi
from skimage.morphology import watershed
import numpy as np
import os
import subprocess
import skimage.io
import tifffile

#-----------------------------

def process_image(image_path, output_path):
    print('image path within function', image_path)
    stack = skimage.io.imread(image_path, is_ome=False)
    #print(stack)
    MAXdata = np.max(stack, axis=0)

    print('writing 1')
    tifffile.imwrite(output_path, MAXdata)

def merge_channels(c0_path, c1_path, output_path):
    c0_image = skimage.io.imread(c0_path)
    c1_image = skimage.io.imread(c1_path)

    # Initialize merged image with zeros
    merged_image = np.zeros((*c0_image.shape, 3), dtype=c0_image.dtype)

    # Assign channels (c0 in magenta, c1 in green)
    merged_image[..., 0] = c0_image  # Red channel for c0
    merged_image[..., 2] = c0_image  # Blue channel for c0
    merged_image[..., 1] = c1_image  # Green channel for c1

    tifffile.imwrite(output_path, merged_image)


# Function to get the number of series in a .lif file
def get_series_count(lif_file, bfconvert_path):
    command = f"{bfconvert_path} -noflat {lif_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout

    # Debugging: Print the output to see what's being captured
    print("bfconvert output:")
    print(output)

    series_count = sum(1 for line in output.split('\n') if line.startswith("Series "))
    return series_count


#----------------------------------------------------------
# Function for Active Contours (Snakes)
def active_contours_segmentation(image, initial_snake):
    # Define parameters for active contours
    snake = segmentation.active_contour(
        image,
        initial_snake,
        alpha=0.015,
        beta=10,
        gamma=0.001
    )
    return snake

# Function for Watershed Segmentation
def watershed_segmentation(image, markers, compactness=0):
    # Compute the gradient of the image for the watershed algorithm
    gradient = filters.sobel(image)
    # Perform watershed segmentation
    labels = watershed(gradient, markers, compactness=compactness)
    return labels

# Function to train Random Forest classifier for segmentation
def train_random_forest_segmentation(features, labels):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(features, labels)
    return classifier

# Function to apply trained Random Forest classifier for segmentation
def apply_classifier(image, classifier, features_func):
    # Extract features for each pixel
    features = features_func(image)
    # Predict class for each pixel based on the classifier
    segmented_image = classifier.predict(features)
    return segmented_image.reshape(image.shape)

# Example function to create features for Random Forest segmentation
def create_features_for_segmentation(image):
    # Compute gradient as a feature
    gradient = filters.sobel(image)
    # Compute other features, e.g., texture, intensity, etc.
    # Stack features into a multi-dimensional array where the last dimension is features
    # This is a placeholder for actual feature computation
    features = np.dstack([gradient])  # Add more features as needed
    # Reshape features to be in the format (num_samples, num_features)
    num_samples = np.prod(image.shape[:2])
    num_features = features.shape[2]
    features = features.reshape((num_samples, num_features))
    return features

