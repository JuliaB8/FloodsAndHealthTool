import rasterio
import numpy as np
import yaml
import dask.array as da
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
import os

# Step 1: Define Flexible Directories
input_dir = r"N:\Projects\11211000\11211459\F. Other information\003 Floods and Health"
output_dir = r"N:\Projects\11211000\11211459\F. Other information\003 Floods and Health\output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 2: Define Input File Paths
population_raster = os.path.join(input_dir, "moz_population_combined_compressed.tif")
urban_rural_raster = os.path.join(input_dir, "Urban_rural_classification.tif")
sanitation_file = os.path.join(input_dir, "sanitation_data.yaml")
gdp_file = os.path.join(input_dir, "gdp_data.yaml")
output_raster = os.path.join(output_dir, "ecoli_emissions2.tif")

# Step 3: Validate Input Files Exist
for file_path in [population_raster, urban_rural_raster, sanitation_file, gdp_file]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"⚠️ Required file not found: {file_path}")

# Standard E. coli Emission Per Person (CFU/100mL)
ECOLI_PER_PERSON = 1e9  

# Load GDP Data & Compute Weight Factor (WF)
with open(gdp_file, "r") as file:
    gdp_data = yaml.safe_load(file)

country_name = input("Enter country name: ").strip()

gdp_per_capita = gdp_data["countries"].get(country_name, {}).get("GDP_per_capita", None)
if gdp_per_capita is None:
    raise ValueError(f"⚠️ GDP data not found for {country_name}")

weight_factor = max(0.5, (1 - (gdp_per_capita / 80)))

# Load Sanitation Data from YAML
sanitation_file = os.path.join(input_dir, "sanitation_data.yaml")

with open(sanitation_file, "r") as file:
    sanitation_data = yaml.safe_load(file)

# Validate the country data exists
if country_name not in sanitation_data["countries"]:
    raise ValueError(f"⚠️ Sanitation data not found for {country_name}")

# Extract sanitation coverage for the selected country
country_coverage = sanitation_data["countries"][country_name]["sanitation_coverage"]

# Extract global reduction factors (same for all countries)
sanitation_types = ["Safe", "Advanced", "Basic", "None"]
urban_reduction_factors = np.array([
    sanitation_data["sanitation_types"][st]["urban_reduction_factor"] for st in sanitation_types
])
rural_reduction_factors = np.array([
    sanitation_data["sanitation_types"][st]["rural_reduction_factor"] for st in sanitation_types
])

# Convert sanitation coverage to NumPy arrays
urban_coverage = np.array([country_coverage[st]["urban"] / 100 for st in sanitation_types])
rural_coverage = np.array([country_coverage[st]["rural"] / 100 for st in sanitation_types])

# Compute weighted sanitation factors
urban_sanitation_factor = np.dot(urban_reduction_factors, urban_coverage)
rural_sanitation_factor = np.dot(rural_reduction_factors, rural_coverage)

# Load Population Raster & Get Extent
with rasterio.open(population_raster) as pop_src:
    pop_transform = pop_src.transform
    pop_bounds = pop_src.bounds  # Get extent of population raster
    pop_crs = pop_src.crs  # Coordinate Reference System
    pop_res = pop_src.res  # Resolution (pixel size)
    pop_shape = (pop_src.height, pop_src.width)  # Shape of raster
    profile = pop_src.profile  # Save raster metadata

# Load & Clip Urban-Rural Raster to Match Population Raster
with rasterio.open(urban_rural_raster) as urb_src:
    # Ensure CRS Matches
    if urb_src.crs != pop_crs:
        raise ValueError("⚠️ CRS mismatch! Urban-rural raster must match population raster.")

    # Use WarpedVRT to Align Urban-Rural Raster to Population Raster
    with WarpedVRT(urb_src, crs=pop_crs, transform=pop_transform, height=pop_shape[0], width=pop_shape[1], resampling=rasterio.enums.Resampling.nearest) as vrt:
        urb_window = from_bounds(*pop_bounds, vrt.transform)
        urban_rural = vrt.read(1, window=urb_window).astype(np.int8)  # Read only clipped area

print(f"Urban-Rural Raster Shape after clipping: {urban_rural.shape}")

# Convert Urban/Rural Classification into Boolean Masks
urban_mask = da.from_array(urban_rural == 1, chunks=(250, 250))
rural_mask = da.from_array(urban_rural == 2, chunks=(250, 250))

# Load Population Raster (Ensure Single Band & Correct Shape)
with rasterio.open(population_raster) as pop_src:
    population = da.from_array(pop_src.read(1).squeeze(), chunks=(250, 250)).astype(np.float64)  # Ensure (height, width)

# Ensure All Arrays Have the Same Shape & Chunking
chunk_size = population.chunks  # Ensure chunking matches population raster
urban_mask = urban_mask.rechunk(chunk_size)
rural_mask = rural_mask.rechunk(chunk_size)

# Parallel Processing: Compute E. coli Emissions
def compute_emissions(pop_chunk, urban_chunk, rural_chunk):
    """Processes a chunk of the raster in parallel using Dask."""
    
    sanitation_factor = da.ones_like(pop_chunk, dtype=np.float64)  # Default = 1
    
    # Ensure urban and rural masks are boolean
    urban_chunk = urban_chunk.astype(bool)
    rural_chunk = rural_chunk.astype(bool)

    # Apply sanitation factors
    sanitation_factor = da.where(urban_chunk, urban_sanitation_factor, sanitation_factor)
    sanitation_factor = da.where(rural_chunk, rural_sanitation_factor, sanitation_factor)

    return pop_chunk * ECOLI_PER_PERSON * sanitation_factor * weight_factor

# Run Processing in Parallel with Dask (Ensure dtype is specified)
emissions = da.map_blocks(
    compute_emissions, 
    population, urban_mask, rural_mask, 
    dtype=np.float64
)

# Force Execution & Compute Results
emissions = emissions.compute()

# Update GeoTIFF Profile for Multiple Bands
profile.update(dtype=rasterio.float64, count=4)  # 4 Bands

# Create Multi-Band Raster Output
with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(population.compute(), 1)  # Band 1: Population Density
    dst.write(urban_rural.astype(np.float64), 2)  # Band 2: Urban/Rural Classification
    dst.write(emissions, 3)  # Band 3: E. coli Emissions
    dst.write((urban_mask * urban_sanitation_factor + rural_mask * rural_sanitation_factor).compute(), 4)  # Band 4: Reduction Factor

print(f"✅ Multi-band E. coli Emission Map saved as {output_raster}")