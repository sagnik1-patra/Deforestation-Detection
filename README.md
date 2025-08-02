 Low-Density Forest Detection using Satellite Imagery
Western Ghats – 2023 | Real-Time Analysis with Multiscale CNN + AIS + CSA


 Objective
To automatically detect low-density forest areas from satellite images using a hybrid deep learning model: Multiscale CNN + Artificial Immune System (AIS) + Crow Search Algorithm (CSA), applied on .tif satellite images.

 Data Source
Satellite Image: Western_Ghats_Forest_2023.tif

Source: Exported from Google Earth Engine

Resolution: 10m (Sentinel-2)

🛠 Pipeline Overview
1. Satellite Image Download (from GEE)
Use the following GEE code to export a .tif of the forest:

js
Copy
Edit
var region = ee.Geometry.Rectangle([74.5, 10.0, 76.5, 12.5]);  // Western Ghats

var image = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(region)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .median()
  .clip(region)
  .select(['B4', 'B8']);  // Red & NIR bands for NDVI

Export.image.toDrive({
  image: image,
  description: 'Western_Ghats_Forest_2023',
  scale: 10,
  region: region,
  fileFormat: 'GeoTIFF'
});
2. NDVI Calculation & CSV Conversion
python
Copy
Edit
import rasterio
import numpy as np
import pandas as pd

tif_path = "C:/Users/sagni/Downloads/Deforestation Detection/Western_Ghats_Forest_2023.tif"

with rasterio.open(tif_path) as src:
    red = src.read(1).astype(float)
    nir = src.read(2).astype(float)
    transform = src.transform

ndvi = (nir - red) / (nir + red + 1e-6)
rows, cols = np.where(~np.isnan(ndvi))

data = {
    "latitude": [],
    "longitude": [],
    "ndvi": []
}

for r, c in zip(rows, cols):
    lon, lat = transform * (c, r)
    data["latitude"].append(lat)
    data["longitude"].append(lon)
    data["ndvi"].append(ndvi[r, c])

df = pd.DataFrame(data)
df.to_csv("forest_ndvi.csv", index=False)
3. Heatmap Visualization
python
Copy
Edit
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("forest_ndvi.csv")
plt.figure(figsize=(10, 8))
sns.kdeplot(
    x=df["longitude"],
    y=df["latitude"],
    weights=1 - df["ndvi"],  # Low NDVI = sparse forest
    cmap="YlOrRd",
    fill=True,
    thresh=0.01
)
plt.title("Low-Density Forest Heatmap (NDVI)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
4. Low Density Forest Detection using CNN + AIS + CSA
python
Copy
Edit
# This block is simplified for README; full code includes patching, training loop, and prediction

from sklearn.cluster import DBSCAN

# Assume df with lat/lon/ndvi loaded
sparse_df = df[df["ndvi"] < 0.3]  # threshold for sparse areas

coords = sparse_df[["longitude", "latitude"]].to_numpy()
clustering = DBSCAN(eps=0.002, min_samples=10).fit(coords)

sparse_df["cluster"] = clustering.labels_

# Show detected zones
clusters = sparse_df["cluster"].unique()
for cluster_id in clusters:
    if cluster_id == -1: continue
    cluster_points = sparse_df[sparse_df["cluster"] == cluster_id]
    lon = cluster_points["longitude"].mean()
    lat = cluster_points["latitude"].mean()
    print(f"Low-Density Forest Detected at: (Lon: {lon:.5f}, Lat: {lat:.5f})")
5. AIS + CSA + Multiscale CNN
AIS: Immune-inspired model selects optimal NDVI patches.

CSA: Swarm search over spatially filtered forest zones.

Multiscale CNN: Detects tree density features at various resolutions (not included fully here).

 Requirements
bash
Copy
Edit
pip install rasterio numpy pandas seaborn matplotlib scikit-learn
 Project Structure
yaml
Copy
Edit
├── forest_ndvi.csv
├── Western_Ghats_Forest_2023.tif
├── Screenshot 2025-08-02 120142.png
├── heatmap_visualization.py
├── ndvi_calculator.py
├── low_density_detector.py
└── README.md
 Results
Location #	Longitude	Latitude
1	75.87036	11.00001
2	75.87274	11.00001
3	75.89003	11.00001
4	75.89152	11.00001
5	75.89489	11.00001

 Future Work
Integrate seasonal NDVI differences for trend analysis

Add LSTM or Transformer layer for time-series NDVI change prediction

Extend model to flood or drought detection
