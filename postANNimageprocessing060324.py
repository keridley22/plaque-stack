
import numpy as np
import tifffile
from skimage.measure import label, regionprops
from scipy.ndimage import binary_closing, generate_binary_structure
import pandas as pd
import os

def load_and_reduce_stack(filename):
    img = tifffile.imread(filename)
    #start_z, end_z = int(0.3 * img.shape[0]), int(0.7 * img.shape[0])
    start_y, end_y = int(0.3 * img.shape[1]), int(0.7 * img.shape[1])
    start_x, end_x = int(0.3 * img.shape[2]), int(0.7 * img.shape[2])
    reduced_img = img[:, start_y:end_y, start_x:end_x]
    reduced_filename = filename.replace('.tiff', '_reduced.tiff')
    tifffile.imwrite(reduced_filename, reduced_img)
    return reduced_img

def join_and_label_objects(img, connectivity=3):
    struct = generate_binary_structure(img.ndim, connectivity) 
    closed_img = binary_closing(img, structure=struct)
    # Use `connectivity` instead of `structure` for skimage.measure.label
    labeled_img = label(closed_img, connectivity=connectivity) 
    return labeled_img

def calculate_surface_area_and_complexity(region):
    a = region.major_axis_length / 2
    b = region.minor_axis_length / 2
    # Here, `c` needs to be defined. Assuming it's the length along the Z-axis divided by 2. This must be obtained differently.
    c = (region.bbox[3] - region.bbox[0]) / 2  # Update this as per your Z-axis measurement method.

    # Calculate surface area using an approximate formula (Knud Thomsen's formula)
    p = 1.6075
    surface_area = 4 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3) ** (1/p)
    complexity = region.area / surface_area

    # Calculate sphericity
    sphericity = (np.pi ** (1.0 / 3.0)) * ((6 * region.area) ** (2.0 / 3.0)) / surface_area
    
    return surface_area, complexity, sphericity

def load_metadata(metadata_file):
    return pd.read_csv(metadata_file, index_col=None)

def get_summary_statistics(properties, genotype, volume_ratio):
    numobjects = len(properties)

    
    total_area = sum(item['area'] for item in properties)
    mean_sphericity = np.mean([prop['sphericity'] for prop in properties])
    mean_solidity = np.mean([prop['solidity'] for prop in properties])
    mean_complexity = np.mean([prop['complexity'] for prop in properties])
  
    return {
        "total_area": total_area,
        "mean_sphericity": mean_sphericity,
        "mean_solidity": mean_solidity,
        "mean_complexity": mean_complexity,
        "volume_ratio": volume_ratio,
        "genotype": genotype,
        "number_of_objects": numobjects
    }

def calculate_morphological_properties(labeled_img):
    regions = regionprops(labeled_img)
    filtered_regions = [r for r in regions if r.area >= 550]
    
    properties = []

    for region in filtered_regions:
        surface_area, complexity, sphericity = calculate_surface_area_and_complexity(region)
        
        properties.append({
            "label": region.label,
            "area": region.area,
            "extent": region.extent,
            "surface_area": surface_area,
            "complexity": complexity,
            "sphericity": sphericity,
            "major_axis_length": region.major_axis_length,
            "minor_axis_length": region.minor_axis_length,
            "bbox": region.bbox,
            "centroid": region.centroid,
            "solidity": region.solidity
            })


    return properties

def update_and_save_summary(folder_path, properties, volume_ratio, genotype, image_name):
    summary = get_summary_statistics(properties, genotype, volume_ratio)
    summary['volume_ratio'] = volume_ratio
    summary['image_name'] = image_name
    summary_df = pd.DataFrame([summary])
    output_csv = os.path.join(folder_path, f"{image_name}_summary.csv")
    summary_df.to_csv(output_csv, index=False)

    return summary

def save_properties(properties, genotype, output_csv):
        df = pd.DataFrame(properties)
        
        df['Genotype'] = genotype
        df.to_csv(output_csv, index=False)

        return df

def mannwhitneyu_test(df, column, group_column):
    from scipy.stats import mannwhitneyu
    group1 = df[df[group_column] == 'WT'][column]
    group2 = df[df[group_column] == 'APP'][column]
    return mannwhitneyu(group1, group2)

def violinplots(df, column, group_column, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.violinplot(x=group_column, y=column, data=df)
    plt.title(title)
    plt.show()
    plt.savefig(f"{column}_violinplot.png")

def find_largest_region(properties):
    # Sorting regions based on 'area' in descending order to get the largest region first
    sorted_regions = sorted(properties, key=lambda x: x['area'], reverse=True)
    largest_region = sorted_regions[0]
    return largest_region

### Step 2: Calculate Distances to the Largest Region


from scipy.spatial.distance import euclidean

def calculate_distances(largest_region, properties):
    largest_centroid = largest_region['centroid']
    distances = [euclidean(largest_centroid, region['centroid']) for region in properties if region != largest_region]
    return distances

### Step 3: Identify Regions That Cluster Around the Largest Region


def find_clustered_regions(largest_region, properties, distance_threshold):
    largest_centroid = largest_region['centroid']

    print(f"Centroid of the largest region: {largest_centroid}")
    clustered_regions = [region for region in properties if region != largest_region 
                         and euclidean(largest_centroid, region['centroid']) < distance_threshold]
    
    #add largest region to clustered regions
    clustered_regions.append(largest_region)
    print(f"Number of clustered regions: {len(clustered_regions)}")

    print(clustered_regions)
    return clustered_regions

def create_combined_mask(labeled_img, clustered_region_labels):
    # Initialize a new mask with the same shape as the original image, filled with zeros
    new_mask = np.zeros(labeled_img.shape, dtype=bool)  # or np.uint8 if you prefer
    
    # For every label that belongs to a clustered region, add it to the new mask
    for label in clustered_region_labels:
        #print(f"Label: {label}")
        #print(f"Number of pixels in the region: {np.sum(labeled_img == label)}")
        #print(f"Number of pixels in the new mask: {np.sum(new_mask)}")  
        new_mask[labeled_img == label] = True  # Mark the region's pixels as part of the combined object
        
    print(f"Number of pixels in the combined object: {np.sum(new_mask)}")
    #save new mask
    
    return new_mask

from skimage.measure import regionprops, label

def calculate_properties_for_combined_object(combined_mask):
    labeled_combined = label(combined_mask)  # Label the combined object



    #print(f"Number of combined objects: {np.max(labeled_combined)}")
    combined_properties = regionprops(combined_mask.astype(int))
    
    # Assuming there's only one object now, so we take the first (and only) item
    if combined_properties:
        prop = combined_properties[0]
        # Calculate and return the properties you're interested in
        area = prop.area
        centroid = prop.centroid
        surface_area, complexity, sphericity = calculate_surface_area_and_complexity(prop)
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        bbox = prop.bbox
        centroid = prop.centroid
        solidity = prop.solidity


        
        return {
            "area": area,
            "extent": prop.extent, # "extent" is a property that's already calculated by regionprops
            "centroid": centroid,
            "surface_area": surface_area,
            "complexity": complexity,
            "sphericity": sphericity,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
            "bbox": bbox,
            "centroid": centroid,
            "solidity": solidity


        }, labeled_combined
    else:
        print("No combined object found")
        return None


### Putting It All Together

def process_image_for_largest_cluster(labeled_img, filename, distance_threshold=200):
    properties = calculate_morphological_properties(labeled_img)
    
    # Finding the largest region
    largest_region = find_largest_region(properties)

    print(f"Area of the largest region: {largest_region['area']}")
    
    # Finding smaller regions clustered around the largest region
    clustered_regions = find_clustered_regions(largest_region, properties, distance_threshold)
    
    clustered_labels = [region['label'] for region in clustered_regions]

    print(f"Labels of clustered regions: {clustered_labels}")
    combined_mask = create_combined_mask(labeled_img, clustered_labels)

    
    combined_properties, labeled_mask = calculate_properties_for_combined_object(combined_mask)

    #save combined mask

    combined_filename = filename.replace('.tiff', '_combined.tiff')
    tifffile.imwrite(combined_filename, labeled_mask)
    
    return combined_properties


def process_image_pair(m04_file, oc_file, folder_path, metadata):
    print(m04_file[:3])
    genotype = metadata.loc[metadata['filename'] == m04_file[:3], 'Genotype'].values[0]
    
    m04_img = load_and_reduce_stack(os.path.join(folder_path, m04_file))
    oc_img = load_and_reduce_stack(os.path.join(folder_path, oc_file))

    m04_labeled = join_and_label_objects(m04_img)
    oc_labeled = join_and_label_objects(oc_img)
    
    m04_properties = calculate_morphological_properties(m04_labeled)
    oc_properties = calculate_morphological_properties(oc_labeled)

    #save properties to csv

    output_csv_m04 = os.path.join(folder_path, f"{m04_file.replace('_M04.tiff', '')}_m04_properties.csv")
    m04_props = save_properties(m04_properties, genotype, output_csv_m04)

    output_csv_oc = os.path.join(folder_path, f"{oc_file.replace('_OC.tiff', '')}_oc_properties.csv")
    oc_props = save_properties(oc_properties, genotype, output_csv_oc)

    m04_combined_properties = process_image_for_largest_cluster(m04_labeled, m04_file)

    oc_combined_properties = process_image_for_largest_cluster(oc_labeled, oc_file)

    combined_csv_m04 = os.path.join(folder_path, f"{m04_file.replace('_M04.tiff', '')}_m04_combined_properties.csv")
    save_properties([m04_combined_properties], genotype, combined_csv_m04)

    combined_csv_oc = os.path.join(folder_path, f"{oc_file.replace('_OC.tiff', '')}_oc_combined_properties.csv")
    save_properties([oc_combined_properties], genotype, combined_csv_oc)

    #append genotype to combined properties
    m04_combined_properties['Genotype'] = genotype
    oc_combined_properties['Genotype'] = genotype

    #append image name to combined properties

    m04_combined_properties['image_name'] = m04_file.replace('_M04.tiff', '')
    oc_combined_properties['image_name'] = oc_file.replace('_OC.tiff', '')
    
    
    m04_volume = np.sum([prop['area'] for prop in m04_properties])
    oc_volume = np.sum([prop['area'] for prop in oc_properties])
    volume_ratio = m04_volume / oc_volume if oc_volume else 0
    
    image_name = m04_file.replace('_M04.tiff', '')
    m04_summary = update_and_save_summary(folder_path, m04_properties, volume_ratio, genotype, image_name)
    oc_summary = update_and_save_summary(folder_path, oc_properties, volume_ratio, genotype, image_name)

    return m04_props, oc_props, m04_combined_properties, oc_combined_properties, m04_summary, oc_summary

def process_folder(folder_path, metadata):
    all_files = [f for f in os.listdir(folder_path) if 'P+' in f and ('_M04.tiff' in f or '_OC.tiff' in f)]
    m04_files = [f for f in all_files if '_M04.tiff' in f]
    
    paired_files = []
    for m04_file in m04_files:
        oc_file = m04_file.replace('_M04.tiff', '_OC.tiff')
        if oc_file in all_files:
            paired_files.append((m04_file, oc_file))
    
    overallm04summaryproperties = []
    overallocsummaryproperties = []

    overallm04combinedproperties = []
    overalloccombinedproperties = []
    for m04_file, oc_file in paired_files:

        mproperties, ocproperties, mcombined, occombined, msummary, ocsummary = process_image_pair(m04_file, oc_file, folder_path, metadata)
        overallm04summaryproperties.append(msummary)
        overallocsummaryproperties.append(ocsummary)
        overallm04combinedproperties.append(mcombined)
        overalloccombinedproperties.append(occombined)


    overallm04summary = pd.DataFrame(overallm04summaryproperties)
    overallocsummary = pd.DataFrame(overallocsummaryproperties)
    overallm04combined = pd.DataFrame(overallm04combinedproperties)
    overalloccombined = pd.DataFrame(overalloccombinedproperties)

    

    # Save the summary statistics to a CSV file

    overallm04summary.to_csv(os.path.join(folder_path, "overall_m04_summary.csv"), index=False)
    overallocsummary.to_csv(os.path.join(folder_path, "overall_oc_summary.csv"), index=False)
    overallm04combined.to_csv(os.path.join(folder_path, "overall_m04_combined.csv"), index=False)
    overalloccombined.to_csv(os.path.join(folder_path, "overall_oc_combined.csv"), index=False)

    

if __name__ == "__main__":
    folder_path = "/Users/katherineridley/Dropbox (UK Dementia Research Institute)/KRidley/PlaqueStack/Masks"
    metadata_file = os.path.join(folder_path, 'metadata.csv')
    metadata = load_metadata(metadata_file)
    process_folder(folder_path, metadata)
    print("Done!")