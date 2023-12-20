import numpy as np
from skimage.measure import label, regionprops
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Function to extract comprehensive region properties
def extract_region_properties(binary_image):
    labeled_image = label(binary_image)
    properties = regionprops(labeled_image)
    
    all_props = []
    for prop in properties:
        region_metrics = {
            'label': prop.label,
            'area': prop.area,
            'perimeter': prop.perimeter,
            'eccentricity': prop.eccentricity,
            'extent': prop.extent,
            'solidity': prop.solidity,
            'centroid': prop.centroid,
            'orientation': prop.orientation,
            'mean_intensity': prop.mean_intensity,
            # Add more properties here as required
        }
        all_props.append(region_metrics)
    return pd.DataFrame(all_props)

# Function to perform statistical analysis
def perform_statistical_analysis(data):
    # Example: ANOVA between different group areas
    # groups should be defined based on some region properties or external factors
    f_val, p_val = stats.f_oneway(data['group1'], data['group2'], data['group3'])
    print("ANOVA results: F=", f_val, ", P=", p_val)

    # Example: Correlation analysis
    correlation_matrix = data.corr()
    print("Correlation matrix:\n", correlation_matrix)

    # Add more statistical tests as required

# Machine learning functions
def train_random_forest(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(classification_report(y_test, y_pred))

    return classifier

# Example function to add new features for machine learning from regionprops
def create_feature_vector(region_properties_dataframe):
    # Convert properties to a feature vector (pandas DataFrame to numpy array)
    # This example assumes that 'label' is not used as a feature for classification
    features = region_properties_dataframe.drop(['label'], axis=1).values
    # You need to provide the corresponding labels for your features
    labels = np.random.choice([0, 1], size=features.shape[0])  # Placeholder for actual labels
    return features, labels
