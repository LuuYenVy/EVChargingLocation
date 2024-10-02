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
    # Tạo một danh sách để lưu dữ liệu cho DataFrame
    data = []
    for node_id in index_mapping:
        # Tạo vector khởi tạo với kích thước giống như ma trận
        initial_vector = np.zeros(transition_matrix.shape[0], dtype=np.float64)

        # Đặt giá trị 1 tại vị trí của node hiện tại trong vector khởi tạo
        node_index = index_mapping[node_id]
        initial_vector[node_index] = 1

        print(f"Chạy thuật toán Markov cho node {node_id} với index {node_index}")

        # Lặp lại quá trình nhân vector với ma trận chuyển tiếp
        current_vector = initial_vector.copy()
        # Chuyển ma trận chuyển tiếp thành ma trận thưa (sparse)

        # Tiến hành các phép toán với ma trận thưa
        current_vector = csr_matrix(initial_vector).T  # Chuyển vector thành ma trận thưa
        for iteration in range(num_iterations):
            current_vector = transition_matrix_sparse @ current_vector
            # Chuyển current_vector thành dạng dense nếu cần thiết
            current_vector_dense = current_vector.toarray().flatten()

            # Lưu node_index và current_vector tại mỗi lần lặp vào data
            data.append({
                'node_id': node_id,
                'node_index': node_index,
                'iteration': iteration + 1,
                'current_vector': current_vector.copy()  # Sao chép vector hiện tại
            })

        print(f"Trạng thái sau {num_iterations} lần lặp: {current_vector}")

    # Tạo DataFrame từ danh sách data
    df = pd.DataFrame(data)
    return df

def create_df_pair(df_result):
    # Danh sách để lưu dữ liệu cho DataFrame mới
    pairs = []

    # Iterating through rows in df_result
    for idx, row in df_result.iterrows():
        # Chuyển đổi current_vector từ csr_matrix thành mảng numpy
        final_vector_csr = row['current_vector']  # Accessing the csr_matrix from the row
        final_vector_dense = final_vector_csr.toarray().flatten()  # Converting to dense numpy array

        # Tìm node có xác suất lớn nhất
        end_node_index = np.argmax(final_vector_dense)

        # Kiểm tra xem xác suất lớn nhất có khác 0 không
        if final_vector_dense[end_node_index] > 0:
            # Thêm vào danh sách các cặp node
            pairs.append({
                'start_node_id': row['node_id'],      # Accessing the node_id from the row
                'start_node_index': row['node_index'], # Accessing the node_index from the row
                'end_node_index': end_node_index       # Saving the index with the highest probability
            })

    # Tạo DataFrame từ danh sách pairs
    df_pair = pd.DataFrame(pairs)
    return df_pair

# Tạo DataFrame với thông tin đầy đủ
def get_node_info(node_id,highway_nodes_combined_dict):
    return highway_nodes_combined_dict.get(node_id, {'x': None, 'y': None, 'matrix_index': None})

def get_end_node_info(matrix_index,highway_nodes_combined_dict):
    for node_id, info in highway_nodes_combined_dict.items():
        if info['matrix_index'] == matrix_index:
            return node_id, info['x'], info['y']
    return None, None, None

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

        # Lấy danh sách osmid của các node
    highway_node_osmids = highway_nodes.index.tolist()
    adjacency_matrix = nx.adjacency_matrix(G, nodelist=highway_node_osmids,weight='weight')
    
        # Bước 1: Tính tổng số kết nối của mỗi điểm (tổng theo hàng)
    degree = np.sum(adjacency_matrix, axis=1)

    # Bước 2: Chia từng giá trị trong ma trận kề cho tổng số kết nối của điểm đó
    # Thay thế giá trị 0 bằng 1 để tránh chia cho 0
    transition_matrix = adjacency_matrix/ degree[:, np.newaxis]

    # Chuyển ma trận chuyển tiếp thưa thành dạng dense (nếu cần)
    transition_matrix_dense = transition_matrix.todense()

    # Đặt các phần tử trên đường chéo thành 0
    np.fill_diagonal(transition_matrix_dense, 0)
    # Khởi tạo vector trạng thái ban đầu (Markov) với giá trị 0 cho tất cả các node
    initial_vector = np.zeros(len(highway_node_osmids))
    
    # Giả sử transition_matrix là ma trận chuyển tiếp
    # Giả sử highway_node_osmids là danh sách các osmid từ ma trận kề
    # Đọc danh sách node từ file .geojson
    geojson_file = 'DataNode/NodeSelected.geojson'

    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Trích xuất danh sách osmid từ file .geojson
    selected_osmids = [feature['properties']['osmid'] for feature in geojson_data['features']]

    # Giả sử highway_node_osmids là danh sách osmid từ ma trận kề
    # Tạo từ điển ánh xạ từ osmid đến chỉ số trong ma trận kề
    osmid_to_index = {osm_id: idx for idx, osm_id in enumerate(highway_node_osmids)}

    # Tìm chỉ số của các node từ danh sách osmid
    indices = [osmid_to_index[osm_id] for osm_id in selected_osmids if osm_id in osmid_to_index]
    index_mapping = dict(zip(selected_osmids, indices))
    # ánh xạ index trong ma trận với id các node đã được chọn
    # Số lần lặp
    num_iterations = 5
    transition_matrix_sparse = csr_matrix(transition_matrix)
    # Chạy hàm và in kết quả
    df_result = run_markov_chain(transition_matrix_dense,transition_matrix_sparse, index_mapping, num_iterations)
    print(df_result)
    df_result.to_csv("makov.csv")
    # Tạo DataFrame cặp từ df_result
    df_pair = create_df_pair(df_result)
    df_pair['end_node_matrix_index']=df_pair['end_node_index']
    # Drop the 'start_node_index' column
    df_pair = df_pair.drop(columns=['end_node_index'])
    df_pair=df_pair.drop(columns=['start_node_index'])
    # Tạo từ điển ánh xạ từ chỉ số ma trận đến node_id
    highway_nodes_dict_index = {index: node_id for index, node_id in enumerate(highway_node_osmids)}

    # Tạo từ điển chứa tọa độ x và y
    highway_nodes_dict_coords = highway_nodes[['x', 'y']].to_dict(orient='index')

    # Kết hợp hai từ điển
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

