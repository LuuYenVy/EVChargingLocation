import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import osmnx as ox
from geopy.distance import great_circle
from shapely.geometry import LineString, Point
import networkx as nx
import geopandas as gpd
import math
from geopy.distance import geodesic
import argparse

# Hàm tính khoảng cách Haversine
def Haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Bán kính Trái đất tính bằng mét
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Khoảng cách tính bằng mét
    return distance

# Hàm tìm node gần nhất
def find_nearest_node(potential_coords, coord_list):
    min_distance = float('inf')
    nearest_node = None
    for coord in coord_list:
        distance = geodesic(coord, potential_coords).meters
        if distance < min_distance:
            min_distance = distance
            nearest_node = coord
    return nearest_node, min_distance

# Hàm chọn ngẫu nhiên các node
def select_random_nodes(place_name,highway_nodes, apartment_coords, office_coords, parking_coords, fuel_stations_coords, supermarket_coords, G, num_nodes=2000, min_distance=700, proximity_threshold=800):
    selected_nodes = []
    selected_node_ids = set()
    node_counter = 1
    iteration_count = 0
    while len(selected_nodes) < num_nodes:
        if iteration_count >= (len(highway_nodes) - 1):
            break
        potential_node = highway_nodes.sample(1)
        potential_node_id = potential_node.index[0]
        if potential_node_id in selected_node_ids:
            continue
        selected_node_ids.add(potential_node_id)
        iteration_count += 1
        potential_coords = (potential_node.geometry.y.values[0], potential_node.geometry.x.values[0])

        valid_node = True
        for selected_node in selected_nodes:
            last_selected_coords = (selected_node.geometry.y.values[0], selected_node.geometry.x.values[0])
            distance = Haversine(last_selected_coords[0], last_selected_coords[1], potential_coords[0], potential_coords[1])
            if not min_distance <= distance:
                valid_node = False
                break

        if valid_node:
            nearest_apartment, dist_apartment = find_nearest_node(potential_coords, apartment_coords)
            nearest_office, dist_office = find_nearest_node(potential_coords, office_coords)
            nearest_supermarket, dist_supermarket = find_nearest_node(potential_coords, supermarket_coords)
            nearest_fuel_stations, dist_fuel_stations = find_nearest_node(potential_coords, fuel_stations_coords)
            nearest_parking, dist_parking = find_nearest_node(potential_coords, parking_coords)

            min_distance_to_poi = min(dist_apartment, dist_office, dist_supermarket, dist_fuel_stations, dist_parking)
            # print(f"Minimum distance to any POI: {min_distance_to_poi} meters")

            if min_distance_to_poi <= proximity_threshold:
                selected_nodes.append(potential_node)
                # print(f"Node {node_counter} được chọn: Kinh độ {potential_coords[1]}, Vĩ độ {potential_coords[0]}")
                node_counter += 1

    nodes_gdf = gpd.GeoDataFrame(pd.concat(selected_nodes), crs=highway_nodes.crs)
    nodes_gdf.to_file('/DataNode/selected_nodes_{place_name}.geojson', driver='GeoJSON')
    return nodes_gdf

def get_geometries(place_name, tags):
    try:
        return ox.geometries_from_place(place_name, tags=tags)
    except Exception as e:
        print(f"Không tìm thấy dữ liệu cho {tags}: {e}")
        return gpd.GeoDataFrame()  # Trả về một GeoDataFrame rỗng nếu không tìm thấy

def main(place_name):
    # Lấy các địa điểm thuộc các loại hình khác nhau (POIs)
    apartment = get_geometries(place_name, tags={'building': 'apartments', 'residential': 'apartments', 'landuse': 'residential'})
    fuel_stations = get_geometries(place_name, tags={'amenity': 'fuel'})
    supermarkets = get_geometries(place_name, tags={'shop': 'supermarket', 'landuse': 'retail', 'building': 'retail', 'amenity': 'marketplace'})
    commercial = get_geometries(place_name, tags={'landuse': 'commercial'})
    official = get_geometries(place_name, tags={'building': 'office'})
    parking = get_geometries(place_name, tags={'amenity': 'parking'})

    G = ox.graph_from_place(place_name, network_type='drive')

    apartment_projected = apartment.to_crs(epsg=3857)
    supermarket_projected = supermarkets.to_crs(epsg=3857)
    commercial_projected = commercial.to_crs(epsg=3857)
    official_projected = official.to_crs(epsg=3857)
    parking_projected = parking.to_crs(epsg=3857)
    fuel_stations_projected = fuel_stations.to_crs(epsg=3857)

    apartment_centroids = apartment_projected.centroid.to_crs(epsg=4326)
    supermarket_centroids = supermarket_projected.centroid.to_crs(epsg=4326)
    commercial_centroids = commercial_projected.centroid.to_crs(epsg=4326)
    official_centroids = official_projected.centroid.to_crs(epsg=4326)
    parking_centroids = parking_projected.centroid.to_crs(epsg=4326)
    fuel_stations_centroids = fuel_stations_projected.centroid.to_crs(epsg=4326)

    apartment_coords = list(apartment_centroids.apply(lambda x: (x.y, x.x)))
    supermarket_coords = list(supermarket_centroids.apply(lambda x: (x.y, x.x)))
    commercial_coords = list(commercial_centroids.apply(lambda x: (x.y, x.x)))
    official_coords = list(official_centroids.apply(lambda x: (x.y, x.x)))
    parking_coords = list(parking_centroids.apply(lambda x: (x.y, x.x)))
    fuel_stations_coords = list(fuel_stations_centroids.apply(lambda x: (x.y, x.x)))

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    highway_edges = edges[edges['highway'].notnull()]

    highway_node_ids = np.unique(np.concatenate([highway_edges.geometry.apply(lambda geom: geom.coords[0]).values,
                                                 highway_edges.geometry.apply(lambda geom: geom.coords[-1]).values]))
    highway_nodes = nodes[nodes.geometry.apply(lambda point: point.coords[0]).isin(highway_node_ids)]

    selected_nodes_gdf = select_random_nodes(place_name,highway_nodes, apartment_coords, commercial_coords, parking_coords, fuel_stations_coords, supermarket_coords, G)

