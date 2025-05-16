intensity_features = [
    "^.*_Mean$",
    "^.*_Median$",
    "^.*_Sum$",
    "^.*_Maximum$",
    "^.*_Minimum$",
    "^.*_StandardDeviation$",
    "^.*_Variance$",
    "^.*_Skewness$",
    "^.*_Kurtosis$",
    "^.*_WeightedElongation$",
    "^.*_WeightedFlatness$",
]

clean_intensity_feature_patterns = [
    "^.*_Mean$",
    "^.*_Sum$",
    "^.*_StandardDeviation$",
    "^.*_Variance$",
    "^.*_Kurtosis$",
    "^.*_Skewness$",
]

stain_patterns = ["^DAPI.1.*$", "^PCNA.0_.*$", "^pH3.1.$"]

density_feature_patterns = [
    "^DELAUNAY:\d+_Count$",
    "^TOUCH:\d+_Count$",
    "^KNNd:\d+_\w+$",
    "^RAD:\d+_Count$",
]

shape_feature_patterns = [
    "^PhysicalSize$",
    "^Roundness$",
    "^Flatness$",
    "^Elongation$",
    "^FeretDiameter$",
    "^Perimeter$",
    "^EquivalentSphericalRadius$",
]

clean_shape_feature_patterns = [
    "^NumberOfPixels$",
    "^Roundness$",
    "^Flatness$",
    "^Elongation$",
    "^FeretDiameter$",
]

position_feature_patterns = ["^Centroid.*$"]

orientation_feature_patterns = ["^PrincipalAxes.*$", "^EquivalentEllipsoidDiameter.*$"]

stage_feature_patterns = ["nucleiRaw3_count", "log2_nuclieRaw3_count", "cycle"]

MINIMAL_CYCLER_FEATURE_NAMES = [
    "EquivalentSphericalRadius",
    "Roundness",
    "FeretDiameter",
    "PCNA.0_Mean",
    "DAPI.1_Kurtosis",
    "pH3.1_Mean",
]

CYCLER_FEATURE_NAMES = [
    "PhysicalSize",
    "Roundness",
    "Flatness",
    "Elongation",
    "FeretDiameter",
    "PCNA.0_Mean",
    "DAPI.1_Skewness",
    "DAPI.1_Kurtosis",
    "pH3.1_Mean",
]

CYCLER_LABEL_FEATURES = [
    "PhysicalSize",
    "Elongation",
    "Flatness",
    "Roundness",
    "FeretDiameter",
    "Perimeter",
    "EquivalentSphericalPerimeter",
    "EquivalentSphericalRadius",
    "EquivalentEllipsoidDiameter-a",
    "EquivalentEllipsoidDiameter-b",
    "EquivalentEllipsoidDiameter-c",
]

CYCLER_INTENSITY_FEATURES = [
    "PCNA.0_Mean",
    "PCNA.0_Variance",
    "PCNA.0_StandardDeviation",
    "PCNA.0_Skewness",
    "PCNA.0_Kurtosis",
    "PCNA.0_Sum",
    "DAPI.1_Mean",
    "DAPI.1_Variance",
    "DAPI.1_StandardDeviation",
    "DAPI.1_Skewness",
    "DAPI.1_Kurtosis",
    "DAPI.1_Sum",
    "pH3.1_Mean",
    "pH3.1_Variance",
    "pH3.1_StandardDeviation",
    "pH3.1_Skewness",
    "pH3.1_Kurtosis",
    "pH3.1_Sum",
]

MORE_CYCLER_FEATURE_NAMES = CYCLER_LABEL_FEATURES + CYCLER_INTENSITY_FEATURES
