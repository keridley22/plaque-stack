# plaque-stack

# Advanced Image Segmentation Pipeline README

## Overview
This README will provide a comprehensive guide for the Advanced Image Segmentation Pipeline, which leverages sophisticated image processing techniques for analyzing and interpreting complex image data. The pipeline is particularly effective for images with high variability, noise, and artifacts.

**Pipeline is currently under construction, this readme is a guide only**

## Requirements
- Python 3.x
- bioformats
- javabridge
- NumPy
- scikit-image
- scikit-learn
- pandas
- matplotlib

## Installation

First, ensure Python 3.11 is installed on your system. Then, install the required Python libraries using pip:

```bash
pip install bioformats javabridge numpy scikit-image scikit-learn pandas matplotlib
```

For advanced visualization capabilities, install `napari`:

```bash
pip install napari[all]
```

## Modules

The pipeline is structured into the following modules:
1. `image_handling.py`: Contains functions for image preprocessing, smoothing, and advanced segmentation techniques.
2. `morphology.py`: Includes morphological operations to refine segmentation such as opening and closing.
3. `analysis.py`: Responsible for extracting region properties, performing statistical analyses, and running machine learning algorithms.
4. `visualization.py`: Provides functions to visualize data and results, including interactive HTML plots and usage of `napari` for 3D viewing.

## Dependencies

Before using the pipeline, ensure that all dependencies are installed:

- `skimage` for image processing functions.
- `napari` for advanced image visualization.
- `numpy` for numerical operations.

## Usage Chronology

1. **Convert `.lif` to `.tif`**: Use the `bfconvert` tool from the command line to convert image files from `.lif` to `.tif` format.
   
   ```bash
   bfconvert -nooverwrite input.lif output_%n.tif
   ```

2. **Image Handling**: Utilize `image_handling.py` for pre-processing, including smoothing with Gaussian filters, and applying advanced segmentation techniques like active contours, watershed segmentation, or machine learning-based segmentation.

3. **Skeletonization**: Perform skeletonization on the pre-processed images to reduce dimensionality and extract the topological essence of the structures.

4. **Region Properties**: Use `analysis.py` to extract properties from the segmented regions, such as area, perimeter, and more complex geometrical and intensity-based measures.

5. **Region Annotation**: Annotate the regions of interest for subsequent analyses or training of machine learning models.

6. **Statistical Analysis**: Apply statistical tests such as ANOVA or correlation analyses to the extracted properties to identify significant differences or relationships.

7. **Machine Learning**: Train and apply machine learning classifiers within `analysis.py` to predict segmentation classes based on image features.

8. **3D Viewer**: Utilize `napari` to view and analyze 3D image stacks and segmentation results in an interactive environment.

9. **Visualization**: Generate and export plots using `visualization.py`, which provides tools for creating interactive HTML plots for publication and presentation purposes.

## Advanced Notes

- Ensure that the javabridge is correctly configured to interface with the bioformats library for file conversions.
- Machine learning models require a training phase with annotated data before they can be applied to new images.
- Visualizations with `napari` can be enhanced with plugins available within the ecosystem.

---

Please replace this line with your data path or specific instructions on how to run the pipeline scripts, including example command lines if applicable.

For more detailed usage instructions, please refer to the documentation within each module, which provides comprehensive information on function parameters and examples.
