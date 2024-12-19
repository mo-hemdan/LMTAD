#%%
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import ast  # To safely evaluate the string to a list

df = pd.read_csv('code/datasets/porto/train.csv')
test_df = pd.read_csv('code/datasets/porto/test.csv')

#%%
from tqdm.contrib.concurrent import process_map

# Apply process_map to convert POLYLINE to LineString objects in parallel
df['POLYLINE'] = process_map(ast.literal_eval, df['POLYLINE'].values, max_workers=10, chunksize=1000)
test_df['POLYLINE'] = process_map(ast.literal_eval, test_df['POLYLINE'].values, max_workers=10, chunksize=1000)

# %%
def is_valid_polyline(coords):
    return isinstance(coords, list) and len(coords) > 1

valid_rows = process_map(is_valid_polyline, df['POLYLINE'].values, max_workers=10, chunksize=1000)
df_cleaned = df[valid_rows].reset_index(drop=True)

valid_rows = process_map(is_valid_polyline, test_df['POLYLINE'].values, max_workers=10, chunksize=1000)
test_df_cleaned = test_df[valid_rows].reset_index(drop=True)

#%%
# Function to convert a list of coordinates to LineString
def coords_to_linestring(coords):
    # return LineString(coords)
    if isinstance(coords, list) and len(coords) > 1:  # Check if valid polyline
        try:
            return LineString(coords)
        except:
            print('START')
            print(coords)
            print('END')
    return None  #/ Handle invalid cases (empty lists, None)

# Apply process_map to convert POLYLINE to LineString objects in parallel
df_cleaned['linestring'] = process_map(coords_to_linestring, df_cleaned['POLYLINE'].values, max_workers=10, chunksize=1000)
test_df_cleaned['linestring'] = process_map(coords_to_linestring, test_df_cleaned['POLYLINE'].values, max_workers=10, chunksize=1000)
#%%
df_cleaned['geometry'] = df_cleaned['linestring']
gdf = gpd.GeoDataFrame(df_cleaned, geometry='geometry')
test_df_cleaned['geometry'] = test_df_cleaned['linestring']
test_gdf = gpd.GeoDataFrame(test_df_cleaned, geometry='geometry')

#%%
from shapely.geometry import box
bounding_box = gdf.total_bounds
minx, miny, maxx, maxy = bounding_box
# Step 2: Split the bounding box into two equal rectangles based on latitude (mid-latitude)
# mid_latitude = (miny + maxy) / 2  # Midpoint in latitude
lat_range = maxy - miny
mid_latitude = miny + 0.302 * lat_range  # 2/3 of the range from miny
# Create the two rectangles (half bounding boxes)
rect1 = box(minx, miny, maxx, mid_latitude)  # Bottom half
rect2 = box(minx, mid_latitude, maxx, maxy)  # Top half
gdf_part1 = gdf[gdf.geometry.intersects(rect1)]
gdf_part2 = gdf[gdf.geometry.intersects(rect2)]
print(gdf_part1.shape, gdf_part2.shape)
test_gdf_part1 = test_gdf[test_gdf.geometry.intersects(rect1)]
test_gdf_part2 = test_gdf[test_gdf.geometry.intersects(rect2)]
test_gdf_part1.shape, test_gdf_part2.shape

#%%
gdf_part1 = gdf_part1.clip(rect1)
gdf_part2 = gdf_part2.clip(rect2)
test_gdf_part1 = test_gdf_part1.clip(rect1)
test_gdf_part2 = test_gdf_part2.clip(rect2)

#%%
# Assuming your GeoDataFrame is named gdf
from shapely.geometry import MultiLineString

def extract_first_linestring(geom):
    if isinstance(geom, MultiLineString):
        geom = geom.geoms[0]  # Extract the first LineString
    return geom

gdf_part1['geometry'] = process_map(extract_first_linestring, gdf_part1['geometry'].values, max_workers=10, chunksize=1000)  # Adjust max_workers as needed
gdf_part2['geometry'] = process_map(extract_first_linestring, gdf_part2['geometry'].values, max_workers=10, chunksize=1000)  # Adjust max_workers as needed
test_gdf_part1['geometry'] = process_map(extract_first_linestring, test_gdf_part1['geometry'].values, max_workers=10, chunksize=1000)  # Adjust max_workers as needed
test_gdf_part2['geometry'] = process_map(extract_first_linestring, test_gdf_part2['geometry'].values, max_workers=10, chunksize=1000)  # Adjust max_workers as needed

# gdf_part1 = convert_gdf_toLINESTRING(gdf_part1)
# gdf_part2 = convert_gdf_toLINESTRING(gdf_part2)
# test_gdf_part1 = convert_gdf_toLINESTRING(test_gdf_part1)
# test_gdf_part2 = convert_gdf_toLINESTRING(test_gdf_part2)
# %%
# Step 3: Filter the GeoDataFrame based on intersection with the rectangles
def linestring_to_string(geometry):
    if geometry.is_empty:
        return "[]"
    try:
        coords = [list(tup) for tup in geometry.coords]
    except:
        print(geometry)
    # coords = list(list_of_lists)
    return str(coords)

# Convert the geometries to string format and prepare data for CSV
gdf_part1['POLYLINE'] = gdf_part1['geometry'].apply(linestring_to_string)
gdf_part2['POLYLINE'] = gdf_part2['geometry'].apply(linestring_to_string)
test_gdf_part1['POLYLINE'] = test_gdf_part1['geometry'].apply(linestring_to_string)
test_gdf_part2['POLYLINE'] = test_gdf_part2['geometry'].apply(linestring_to_string)

# %%
gdf_part1.drop(columns=['geometry', 'linestring'], inplace=True)
gdf_part2.drop(columns=['geometry', 'linestring'], inplace=True)
test_gdf_part1.drop(columns=['geometry', 'linestring'], inplace=True)
test_gdf_part2.drop(columns=['geometry', 'linestring'], inplace=True)
# %%
gdf_part1.to_csv('code/datasets/porto/train_1.csv', index=False)
gdf_part2.to_csv('code/datasets/porto/train_2.csv', index=False)
test_gdf_part1.to_csv('code/datasets/porto/test_1.csv', index=False)
test_gdf_part2.to_csv('code/datasets/porto/test_2.csv', index=False)
# %%
