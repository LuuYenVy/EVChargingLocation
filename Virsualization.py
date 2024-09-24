import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
from geopy.distance import great_circle
from shapely.geometry import LineString, Point

# Function to plot coordinates on the map
def plot_coords_on_map(coords, title, place_name):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the base map
    gdf_map = ox.geocode_to_gdf(place_name)
    gdf_map.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot the coordinates
    for coord in coords:
        plt.plot(coord[1], coord[0], 'ro', markersize=8)  # latitude, longitude

    # Add title and labels
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the map
    plt.show()

# Hàm chính
def main(place_name):
    # Tải dữ liệu giao thông
    G = ox.graph_from_place(place_name, network_type='drive')  # lấy dữ liệu giao thông có thể đi bằng xe hơi

    # Lấy các node và edge trên tuyến đường highway
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # Lọc các edge là đường highway
    highway_edges = edges[edges['highway'].notnull()]

    # Lấy các node đầu và cuối từ cột geometry của mỗi cạnh
    highway_node_ids = np.unique(np.concatenate([highway_edges.geometry.apply(lambda geom: geom.coords[0]).values,
                                                 highway_edges.geometry.apply(lambda geom: geom.coords[-1]).values]))

    # Chuyển đổi tọa độ node thành chỉ số để khớp với DataFrame nodes
    highway_nodes = nodes[nodes.geometry.apply(lambda point: point.coords[0]).isin(highway_node_ids)]

    # Lấy các địa điểm thuộc các loại hình khác nhau (POIs)
    def get_geometries(place_name, tags):
        try:
            return ox.geometries_from_place(place_name, tags=tags)
        except Exception as e:
            print(f"Không tìm thấy dữ liệu cho {tags}: {e}")
            return gpd.GeoDataFrame()  # Trả về một GeoDataFrame rỗng nếu không tìm thấy

    apartment = get_geometries(place_name, tags={'building': 'apartments', 'residential': 'apartments', 'landuse': 'residential'})
    fuel_stations = get_geometries(place_name, tags={'amenity': 'fuel'})
    supermarkets = get_geometries(place_name, tags={'shop': 'supermarket', 'landuse': 'retail', 'building': 'retail', 'amenity': 'marketplace'})
    commercial = get_geometries(place_name, tags={'landuse': 'commercial'})
    official = get_geometries(place_name, tags={'building': 'office'})
    parking = get_geometries(place_name, tags={'amenity': 'parking'})

    # Chuyển đổi GeoDataFrame về EPSG:3857
    apartment_projected = apartment.to_crs(epsg=3857)
    supermarket_projected = supermarkets.to_crs(epsg=3857)
    commercial_projected = commercial.to_crs(epsg=3857)
    official_projected = official.to_crs(epsg=3857)
    parking_projected = parking.to_crs(epsg=3857)
    fuel_stations_projected = fuel_stations.to_crs(epsg=3857)

    # Tính centroid trong hệ tọa độ phẳng
    apartment_centroids = apartment_projected.centroid
    supermarket_centroids = supermarket_projected.centroid
    commercial_centroids = commercial_projected.centroid
    official_centroids = official_projected.centroid
    parking_centroids = parking_projected.centroid
    fuel_stations_centroids = fuel_stations_projected.centroid

    # Chuyển centroid về hệ tọa độ địa lý (EPSG:4326)
    apartment_centroids = apartment_centroids.to_crs(epsg=4326)
    supermarket_centroids = supermarket_centroids.to_crs(epsg=4326)
    commercial_centroids = commercial_centroids.to_crs(epsg=4326)
    official_centroids = official_centroids.to_crs(epsg=4326)
    parking_centroids = parking_centroids.to_crs(epsg=4326)
    fuel_stations_centroids = fuel_stations_centroids.to_crs(epsg=4326)

    # Trích xuất tọa độ và lưu vào các list
    apartment_coords = list(apartment_centroids.apply(lambda x: (x.y, x.x)))
    supermarket_coords = list(supermarket_centroids.apply(lambda x: (x.y, x.x)))
    commercial_coords = list(commercial_centroids.apply(lambda x: (x.y, x.x)))
    official_coords = list(official_centroids.apply(lambda x: (x.y, x.x)))
    parking_coords = list(parking_centroids.apply(lambda x: (x.y, x.x)))
    fuel_stations_coords = list(fuel_stations_centroids.apply(lambda x: (x.y, x.x)))

    # Plotting the maps
    plot_coords_on_map(apartment_coords, "Apartment Locations", place_name)
    plot_coords_on_map(supermarket_coords, "Supermarket Locations", place_name)
    plot_coords_on_map(commercial_coords, "Commercial Area Locations", place_name)
    plot_coords_on_map(official_coords, "Official Area Locations", place_name)
    plot_coords_on_map(parking_coords, "Parking Area Locations", place_name)
    plot_coords_on_map(fuel_stations_coords, "Fuel Station Locations", place_name)


