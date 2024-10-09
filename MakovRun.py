import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import osmnx as ox
from geopy.distance import great_circle
from shapely.geometry import LineString, Point
import networkx as nx
import scipy.sparse as sp
import json
import scipy.sparse as sp
from scipy.sparse import coo_matrix, csr_matrix
import argparse

def run_markov_chain(transition_matrix,transition_matrix_sparse, index_mapping, num_iterations):
    data = []
    for node_id in index_mapping:
        initial_vector = np.zeros(transition_matrix.shape[0], dtype=np.float64)

        node_index = index_mapping[node_id]
        initial_vector[node_index] = 1

        print(f"Chạy thuật toán Markov cho node {node_id} với index {node_index}")

        current_vector = initial_vector.copy()
        current_vector = csr_matrix(initial_vector).T  
        for iteration in range(num_iterations):
            current_vector = transition_matrix_sparse @ current_vector
            current_vector_dense = current_vector.toarray().flatten()

            data.append({
                'node_id': node_id,
                'node_index': node_index,
                'iteration': iteration + 1,
                'current_vector': current_vector.copy() 
            })

        print(f"Trạng thái sau {num_iterations} lần lặp: {current_vector}")

    df = pd.DataFrame(data)
    return df

def create_df_pair(df_result):
    pairs = []
    used_end_nodes = set()
    nodes_added = 0

    for idx, row in df_result.iterrows():
        final_vector_csr = row['current_vector']  
        final_vector_dense = final_vector_csr.toarray().flatten()  

        end_node_index = np.argmax(final_vector_dense)

        if final_vector_dense[end_node_index] > 0:
            if end_node_index != row['node_index'] and end_node_index not in used_end_nodes:
                pairs.append({
                    'start_node_id': row['node_id'],      
                    'start_node_index': row['node_index'], 
                    'end_node_index': end_node_index       
                })
                
                used_end_nodes.add(end_node_index)
                nodes_added += 1  

            else:
                candidates = np.argsort(final_vector_dense)[::-1]  
                for candidate_index in candidates:
                    if (final_vector_dense[candidate_index] > 0 and
                        candidate_index != row['node_index'] and
                        candidate_index not in used_end_nodes):
                        pairs.append({
                            'start_node_id': row['node_id'],
                            'start_node_index': row['node_index'],
                            'end_node_index': candidate_index
                        })
                        used_end_nodes.add(candidate_index)
                        nodes_added += 1  
                        break
                else:
                    continue
        else:
            continue
    df_pair = pd.DataFrame(pairs)
    return df_pair


def get_node_info(node_id,highway_nodes_combined_dict):
    return highway_nodes_combined_dict.get(node_id, {'x': None, 'y': None, 'matrix_index': None})

def get_end_node_info(matrix_index,highway_nodes_combined_dict):
    for node_id, info in highway_nodes_combined_dict.items():
        if info['matrix_index'] == matrix_index:
            return node_id, info['x'], info['y']
    return None, None, None

def main(place_name):
    G = ox.graph_from_place(place_name, network_type='drive')  

    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

  
    highway_edges = edges[edges['highway'].notnull()]

   
    highway_node_ids = np.unique(np.concatenate([highway_edges.geometry.apply(lambda geom: geom.coords[0]).values,
                                                 highway_edges.geometry.apply(lambda geom: geom.coords[-1]).values]))

    highway_nodes = nodes[nodes.geometry.apply(lambda point: point.coords[0]).isin(highway_node_ids)]

    def get_geometries(place_name, tags):
        try:
            return ox.geometries_from_place(place_name, tags=tags)
        except Exception as e:
            print(f"Không tìm thấy dữ liệu cho {tags}: {e}")
            return gpd.GeoDataFrame() 

    apartment = get_geometries(place_name, tags={'building': 'apartments', 'residential': 'apartments', 'landuse': 'residential'})
    fuel_stations = get_geometries(place_name, tags={'amenity': 'fuel'})
    supermarkets = get_geometries(place_name, tags={'shop': 'supermarket', 'landuse': 'retail', 'building': 'retail', 'amenity': 'marketplace'})
    commercial = get_geometries(place_name, tags={'landuse': 'commercial'})
    official = get_geometries(place_name, tags={'building': 'office'})
    parking = get_geometries(place_name, tags={'amenity': 'parking'})

    apartment_projected = apartment.to_crs(epsg=3857)
    supermarket_projected = supermarkets.to_crs(epsg=3857)
    commercial_projected = commercial.to_crs(epsg=3857)
    official_projected = official.to_crs(epsg=3857)
    parking_projected = parking.to_crs(epsg=3857)
    fuel_stations_projected = fuel_stations.to_crs(epsg=3857)

    apartment_centroids = apartment_projected.centroid
    supermarket_centroids = supermarket_projected.centroid
    commercial_centroids = commercial_projected.centroid
    official_centroids = official_projected.centroid
    parking_centroids = parking_projected.centroid
    fuel_stations_centroids = fuel_stations_projected.centroid

    apartment_centroids = apartment_centroids.to_crs(epsg=4326)
    supermarket_centroids = supermarket_centroids.to_crs(epsg=4326)
    commercial_centroids = commercial_centroids.to_crs(epsg=4326)
    official_centroids = official_centroids.to_crs(epsg=4326)
    parking_centroids = parking_centroids.to_crs(epsg=4326)
    fuel_stations_centroids = fuel_stations_centroids.to_crs(epsg=4326)

    highway_node_osmids = highway_nodes.index.tolist()
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=highway_node_osmids,weight='weight')
    
    degree = np.sum(adjacency_matrix, axis=1)

    transition_matrix = adjacency_matrix/ degree[:, np.newaxis]

    transition_matrix_dense = transition_matrix.todense()


    np.fill_diagonal(transition_matrix_dense, 0)

    initial_vector = np.zeros(len(highway_node_osmids))

    geojson_file = 'DataNode/NodeSelected.geojson'

    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    # Kiểm tra độ dài của geojson_data (số lượng features)
    length_of_geojson_data = len(geojson_data['features'])

    # In ra độ dài
    print(f"Tổng số Node: {length_of_geojson_data}")


    selected_osmids = [feature['properties']['osmid'] for feature in geojson_data['features']]

    osmid_to_index = {osm_id: idx for idx, osm_id in enumerate(highway_node_osmids)}

    indices = [osmid_to_index[osm_id] for osm_id in selected_osmids if osm_id in osmid_to_index]
    index_mapping = dict(zip(selected_osmids, indices))
    num_iterations = 5
    transition_matrix_sparse = csr_matrix(transition_matrix)

    df_result = run_markov_chain(transition_matrix_dense,transition_matrix_sparse, index_mapping, num_iterations)
    print(df_result)
    df_result.to_csv("makov.csv")

    df_pair = create_df_pair(df_result)
    df_pair['end_node_matrix_index']=df_pair['end_node_index']

    df_pair = df_pair.drop(columns=['end_node_index'])
    df_pair=df_pair.drop(columns=['start_node_index'])

    highway_nodes_dict_index = {index: node_id for index, node_id in enumerate(highway_node_osmids)}

    highway_nodes_dict_coords = highway_nodes[['x', 'y']].to_dict(orient='index')
    highway_nodes_combined_dict = {
        node_id: {
            'matrix_index': index,
            'x': coords['x'],
            'y': coords['y']
        }
        for index, node_id in highway_nodes_dict_index.items()
        if node_id in highway_nodes_dict_coords
        for coords in [highway_nodes_dict_coords[node_id]]
    }

    data = []
    for _, row in df_pair.iterrows():
        start_node_id = row['start_node_id']
        end_node_matrix_index = row['end_node_matrix_index']

        start_info = get_node_info(start_node_id,highway_nodes_combined_dict)
        end_node_id, end_x, end_y = get_end_node_info(end_node_matrix_index,highway_nodes_combined_dict)

        if start_info and end_node_id:
            data.append({
                'start_node_id': start_node_id,
                'start_x': start_info['x'],
                'start_y': start_info['y'],
                'end_node_id': end_node_id,
                'end_x': end_x,
                'end_y': end_y
            })
    df_nodes_info = pd.DataFrame(data)
    df_nodes_info.to_csv("Node_pair.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("place_name", help="Tên địa điểm để tìm kiếm.")
    args = parser.parse_args()
    main(args.place_name)

