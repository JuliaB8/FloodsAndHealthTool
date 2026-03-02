import os
import requests
import rasterio
import numpy as np
from rasterio.enums import Resampling
from osgeo import gdal 
import time

# Define parameters
COUNTRY = "MOZ"
YEAR = 2020
OUTPUT_DIR = r"C:\FloodsAndHealthTool-main\examples\Data bronnen\worldpop_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Age Bins from WorldPop
CHILD_AGES = ["0", "1", "5"] # 0-10 years (Children)
ADULT_AGES = ["10", "15", "20", "25", "30", "35", "40", 
              "45", "50", "55", "60", "65", "70", "75", "80"]  # 10+ years (Adults)

#  ✅ Fix: Correct WorldPop URL structure
BASE_URL = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020_Constrained/{year}/{country}/"

# Function to download WorldPop data
def download_worldpop_data(country, gender, age_bin, year, output_dir):
    """Downloads WorldPop raster data for a given country, gender, and age group."""
    url = f"{BASE_URL}{country.lower()}_{gender}_{age_bin}_{year}_constrained.tif".format(
        year=year, country=country
    )
    output_path = os.path.join(output_dir, f"{country.lower()}_{gender}_{age_bin}_{year}_constrained.tif")
    
    if not os.path.exists(output_path):  # Avoid re-downloading
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Downloaded: {output_path}")
        else:
            print(f"⚠️ File not found: {url}. Skipping...")
            return None
    return output_path

# Function to load and preprocess raster data
def load_raster(file_path):
    """Loads raster data and sets negative values to zero."""
    if file_path is None:  # Skip if file doesn't exist
        return None, None
    with rasterio.open(file_path) as src:
        data = src.read(1, resampling=Resampling.bilinear)
        data[data < 0] = 0  # Set negative values to zero
        profile = src.profile
    return data, profile

# Function to merge population data across valid age bins
def merge_population_data(ages, gender1, gender2):
    """Merges male and female population data for a given age range."""
    merged_data = None
    profile = None

    for age_bin in ages:
        m_path = download_worldpop_data(COUNTRY, gender1, age_bin, YEAR, OUTPUT_DIR)
        f_path = download_worldpop_data(COUNTRY, gender2, age_bin, YEAR, OUTPUT_DIR)
        
        m_data, profile = load_raster(m_path)
        f_data, _ = load_raster(f_path)

        if m_data is None or f_data is None:
            continue  # Skip if file missing
        
        if merged_data is None:
            merged_data = m_data + f_data  # Initialize array
        else:
            merged_data += (m_data + f_data)  # Sum up population
    
    return merged_data, profile

# Now processing only the correct age bins
print("\n🔄 Processing Children (0-10 years)...")
children_data, profile = merge_population_data(CHILD_AGES, "m", "f")

print("\n🔄 Processing Adults (10+ years)...")
adults_data, _ = merge_population_data(ADULT_AGES, "m", "f")

# Ensure the profile always has 3 bands
profile.update(count=3, dtype=rasterio.float32)

# Initialize total_population even if one dataset is missing
if children_data is None:
    children_data = np.zeros_like(adults_data, dtype=np.float32)
if adults_data is None:
    adults_data = np.zeros_like(children_data, dtype=np.float32)

total_population = children_data + adults_data  # Ensure third band exists

# Debugging print to check array shapes
print(f"Children shape: {children_data.shape}, dtype: {children_data.dtype}")
print(f"Adults shape: {adults_data.shape}, dtype: {adults_data.dtype}")
print(f"Total shape: {total_population.shape}, dtype: {total_population.dtype}")

# Define output path
OUTPUT_RASTER = os.path.join(OUTPUT_DIR, f"{COUNTRY.lower()}_population_combined.tif")

# Write the raster with three bands
with rasterio.open(OUTPUT_RASTER, 'w', **profile) as dst:
    dst.write_band(1, children_data.astype(np.float32))  # Band 1: Children
    dst.write_band(2, adults_data.astype(np.float32))    # Band 2: Adults
    dst.write_band(3, total_population.astype(np.float32))  # Band 3: Total Population

print(f"\n✅ Final processed raster saved at: {OUTPUT_RASTER}")

# Verify the number of bands in the output raster
with rasterio.open(OUTPUT_RASTER) as src:
    print(f"Raster bands available: {src.count}")
    for i in range(1, src.count + 1):
        print(f"Band {i} min/max: {src.read(i).min()} / {src.read(i).max()}")


# Re-save the TIFF file with compression
COMPRESSED_RASTER = os.path.abspath(os.path.join(OUTPUT_DIR, f"{COUNTRY.lower()}_population_combined_compressed.tif"))

with rasterio.open(OUTPUT_RASTER) as src:
    profile = src.profile
    profile.update(driver='GTiff', dtype=rasterio.float32, compress='deflate')  # Explicitly set driver

    # Write the compressed raster
    with rasterio.open(COMPRESSED_RASTER, 'w', **profile) as dst:
        for band_id in range(1, src.count + 1):
            dst.write(src.read(band_id).astype(np.float32), band_id)

# Confirm the compressed file exists before proceeding
if not os.path.exists(COMPRESSED_RASTER):
    raise FileNotFoundError(f"❌ ERROR: The compressed raster file was not created: {COMPRESSED_RASTER}")

print(f"\n✅ Compressed raster saved at: {COMPRESSED_RASTER}")

time.sleep(2)

def clip_population_to_flood(flood_tif, pop_tif, output_tif):
    """Clips the population raster to match the flood raster's extent while keeping all bands."""
    if not os.path.exists(pop_tif):
        raise FileNotFoundError(f"❌ ERROR: The population raster does not exist: {pop_tif}")

    if not os.path.exists(flood_tif):
        raise FileNotFoundError(f"❌ ERROR: The flood raster does not exist: {flood_tif}")

    print(f"✅ Clipping population raster: {pop_tif}")

    # Open the flood raster (template)
    flood_ds = gdal.Open(flood_tif)
    flood_geo = flood_ds.GetGeoTransform()
    flood_proj = flood_ds.GetProjection()
    x_size = flood_ds.RasterXSize
    y_size = flood_ds.RasterYSize

    # Open the population raster
    pop_ds = gdal.Open(pop_tif)
    if pop_ds is None:
        raise RuntimeError(f"❌ ERROR: Failed to open the population raster: {pop_tif}")

    # Clip the population raster to the flood extent
    clipped_pop = gdal.Translate(
        output_tif, pop_ds,
        projWin=(flood_geo[0], flood_geo[3], 
                 flood_geo[0] + x_size * flood_geo[1], 
                 flood_geo[3] + y_size * flood_geo[5]),  # Match flood extent
        projWinSRS=flood_proj,  # Use flood raster's projection
        format='GTiff'
    )

    # Ensure the file was created successfully
    if not os.path.exists(output_tif):
        raise RuntimeError(f"❌ ERROR: Failed to create clipped raster: {output_tif}")

    # Close datasets
    clipped_pop = None
    flood_ds = None
    pop_ds = None

    print(f"✅ Clipped population raster saved as {output_tif}")

# Ensure we use the correct absolute path for the compressed raster
COMPRESSED_RASTER = os.path.abspath(os.path.join(OUTPUT_DIR, f"{COUNTRY.lower()}_population_combined_compressed.tif"))
CLIPPED_RASTER = os.path.abspath(os.path.join(OUTPUT_DIR, "clipped_population.tif"))
FLOOD_RASTER = os.path.abspath('C:/FloodsAndHealthTool-main/examples/Data bronnen/flooded_zoom.tif')

# Verify that the compressed raster exists before proceeding
if not os.path.exists(COMPRESSED_RASTER):
    raise FileNotFoundError(f"❌ ERROR: The compressed raster file was not found: {COMPRESSED_RASTER}")

# Run the clipping function
clip_population_to_flood(FLOOD_RASTER, COMPRESSED_RASTER, CLIPPED_RASTER)