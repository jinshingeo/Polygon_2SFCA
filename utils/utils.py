import geopandas as gpd
import pandas as pd
import osmnx as ox
import time
import numpy as np
from tqdm import tqdm, trange
from shapely.geometry import Point, MultiPoint, Polygon
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
#from shapely.ops import cascaded_union, unary_union
import utils
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Union

def nearest_osm(network, gdf):
    for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        if row.geometry.geom_type == 'Point':
            nearest_osm = ox.distance.nearest_nodes(network, X=row.geometry.x, Y=row.geometry.y)
        elif row.geometry.geom_type =='Polygon' or row.geometry.geom_type =='MultiPolygon':
            nearest_osm = ox.distance.nearest_nodes(network, X=row.geometry.centroid.x, Y=row.geometry.centroid.y)
        else:
            print(row.geometry.geom_type)
            continue

        gdf.loc[idx, 'nearest_osm'] = nearest_osm
    
    return gdf

def network_settings_drive(network):
    '''자동차 네트워크 설정'''
    for u, v, data in network.edges(data=True):
        if 'maxspeed' in data.keys():
            speed_type = type(data['maxspeed'])
            if (speed_type==str):
                data['maxspeed']=float(data['maxspeed'].split()[0])
            else:
                data['maxspeed']=float(data['maxspeed'][0].split()[0])

        else:
            data['maxspeed']= 60


            
            temp_speed = data['maxspeed'][0] if isinstance(data['maxspeed'], list) else data['maxspeed']# temp_speed가 문자열인 경우만 split 사용
            if isinstance(temp_speed, str):
                temp_speed = temp_speed.split(' ')[0]   
        data['maxspeed_meters'] = data['maxspeed'] * 16.6667  # km/h -> m/s * 0.27778 , km/h -> m/min * 16.6667
        data['time']= float(data['length']/data['maxspeed_meters'])


    for node, data in network.nodes(data=True):
        data['geometry'] = Point(data['x'],data['y'])
    
    print("Drive network set done")
    
    return network

def network_settings_walk(network):
    """보행자 네트워크 설정"""
    walking_speed = 4.644  # km/h (일반적인 보행 속도 4km/h) [3]
    
    for u, v, data in network.edges(data=True):
        data['maxspeed'] = walking_speed
        data['maxspeed_meters'] = walking_speed * 16.6667  # km/h -> m/min
        data['time'] = float(data['length']/data['maxspeed_meters'])
    
    # 노드 geometry 정보 추가
    for node, data in network.nodes(data=True):
        data['geometry'] = Point(data['x'], data['y'])
    
    print("walk network set done")
    return network

from typing import Dict

def step1_E2SFCA(
    weights: Dict[Union[float, int],Union[float, int]],  # 키: 임계값(시간/거리), 값: 가중치
    supply: gpd.GeoDataFrame,
    supply_attr: str,
    demand: gpd.GeoDataFrame,
    demand_attr: str,
    network: nx.MultiDiGraph
):
    supply_ = supply.copy(deep=True)
    supply_['ratio'] = 0

    
    for i in tqdm(range(supply_.shape[0])):
        total_demand = 0
        prev_nodes = set()  # 이전 시간 구간 노드 누적
        
        # 거리 오름차순으로 정렬 (5 → 10 → 15)
        for time, weight in sorted(weights.items(), key=lambda x: x[0]):
            # 현재 거리까지의 모든 노드 계산
            temp_nodes = nx.single_source_dijkstra_path_length(network, supply_.loc[i, 'nearest_osm'], cutoff=time, weight='time'
            ).keys()
            
            # 현재 구간 노드 = 전체 노드 - 이전 구간 노드
            current_nodes = set(temp_nodes) - prev_nodes
            
            # 수요 계산 및 가중치 적용
            demand_sum = demand.loc[demand['nearest_osm'].isin(current_nodes), demand_attr].sum() * weight
            
            total_demand += demand_sum
            
            # 다음 구간을 위해 노드 업데이트
            prev_nodes.update(temp_nodes)
        
        # 최종 ratio 계산
        supply_value = supply_.loc[i, supply_attr]
        step1_ratio = (supply_value / total_demand) * 100000
        supply_.loc[i, 'ratio'] = step1_ratio
        
    return supply_


def step2_E2SFCA(
    weights: Dict[Union[float, int],Union[float, int]],  # 키: 임계값(시간/거리), 값: 가중치,
    result_step1: pd.DataFrame,
    demand: pd.DataFrame,
    network: nx.Graph
) -> pd.DataFrame:
    """
    E2SFCA 방법론의 두 번째 단계를 수행하여 수요지점의 접근성 지수를 계산합니다.

    Args:
        weights (Dict[int, float]): 시간 임계값(분)과 가중치 쌍 (예: {5: 0.5, 10: 0.3})
        result_step1 (pd.DataFrame): 1단계에서 생성된 공급 시설 ratio 데이터프레임
        demand (pd.DataFrame): 수요지점 정보 (반드시 'nearest_osm' 컬럼 포함)
        network (nx.Graph): 이동 시간 계산을 위한 도로 네트워크 그래프

    Returns:
        pd.DataFrame: 'access' 컬럼이 추가된 수요지점 데이터프레임

    Note:
        - Dijkstra 알고리즘을 사용한 이동 시간 기반 접근성 계산
        - 누적 거리 구간 적용 (5분->10분->15분 점진적 확장)
    """
    demand_ = demand.copy(deep=True)
    demand_['access'] = 0
    
    for z in tqdm(range(demand_.shape[0]), desc="Processing demand points"):
        total_sum = 0
        prev_nodes: Set[int] = set()
        
        # 거리 구간별 가중치 적용 (오름차순 정렬)
        for time, weight in sorted(weights.items(), key=lambda x: x[0]):
            # 현재 시간 임계값까지의 모든 노드 탐색
            temp_nodes = nx.single_source_dijkstra_path_length(
                network,
                source=demand_.loc[z, 'nearest_osm'],
                cutoff=time,
                weight='time'
            ).keys()
            
            # 현재 구간 노드 = 전체 노드 - 이전 구간 노드
            current_nodes = set(temp_nodes) - prev_nodes
            
            # 공급 시설 ratio 합산 (무한대 값 제외)
            sum_ratio = result_step1.loc[
                result_step1['nearest_osm'].isin(current_nodes), 
                'ratio'
            ].replace([np.inf, -np.inf], np.nan).dropna().sum() * weight
            
            total_sum += sum_ratio
            prev_nodes.update(temp_nodes)
        
        demand_.loc[z, 'access'] = total_sum

    return demand_

########### 그리드를 고려한 면적 형태 접근성 분석 코드 #############

# def create_polygon_grid(polygon, grid_size=200):
#     """
#     폴리곤을 정규 격자로 분할하여 각 격자의 센트로이드를 반환
    
#     Args:
#         polygon: shapely.geometry.Polygon 객체
#         grid_size: 격자 크기 (미터 단위)
    
#     Returns:
#         list: 격자 센트로이드들의 Point 객체 리스트
#     """
#     from shapely.geometry import box, Point
#     import numpy as np
    
#     # 폴리곤의 경계 좌표 획득
#     minx, miny, maxx, maxy = polygon.bounds
    
#     # 격자 생성을 위한 좌표 배열
#     cols = list(np.arange(minx, maxx + grid_size, grid_size))
#     rows = list(np.arange(miny, maxy + grid_size, grid_size))
    
#     grid_centroids = []
    
#     for x in cols[:-1]:
#         for y in rows[:-1]:
#             # 격자 셀 생성
#             grid_cell = box(x, y, x + grid_size, y + grid_size)
            
#             # 원본 폴리곤과 교차하는 격자만 선택
#             if polygon.intersects(grid_cell):
#                 # 교차 영역의 센트로이드 계산
#                 intersection = polygon.intersection(grid_cell)
#                 if not intersection.is_empty:
#                     grid_centroids.append(intersection.centroid)
    
#     return grid_centroids

# create_polygon_grid 함수 수정
from shapely.geometry import box
def create_polygon_grid(polygon, grid_size):
    """격자 생성 (미터 단위 좌표계 필수)"""
    # 경계 계산
    minx, miny, maxx, maxy = polygon.bounds
    
    # 격자 좌표 배열 (미터 단위)
    x_coords = np.arange(minx, maxx + grid_size, grid_size)
    y_coords = np.arange(miny, maxy + grid_size, grid_size)
    
    grids = []
    for x in x_coords[:-1]:  # 마지막 격자 제외
        for y in y_coords[:-1]:
            cell = box(x, y, x+grid_size, y+grid_size)
            if polygon.intersects(cell):
                intersection = polygon.intersection(cell)
                if not intersection.is_empty:
                    grids.append(intersection.centroid)
    return grids
    
def explode_grids(gdf, grid_size):
    """격자 분할 및 레코드 증식 (수정 버전)"""
    exploded = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom.is_empty or not geom.is_valid:
            continue
            
        # 폴리곤 처리
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    grids = create_polygon_grid(poly, grid_size)
                    for pt in grids:
                        new_row = row.copy()
                        new_row.geometry = pt
                        new_row['parent_id'] = idx  # parent_id 추가
                        exploded.append(new_row)
            else:
                grids = create_polygon_grid(geom, grid_size)
                for pt in grids:
                    new_row = row.copy()
                    new_row.geometry = pt
                    new_row['parent_id'] = idx  # parent_id 추가
                    exploded.append(new_row)
        else:
            # 점/선형 데이터 처리
            new_row = row.copy()
            new_row['parent_id'] = idx
            exploded.append(new_row)
    
    return gpd.GeoDataFrame(exploded, crs=gdf.crs)

# def nearest_osm_enhanced(network, gdf, grid_size=200):
#     """
#     개선된 nearest_osm 함수 - 폴리곤의 경우 격자 센트로이드 기반
#     """
#     for idx, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
#         if row.geometry.geom_type == 'Point':
#             nearest_osm = ox.distance.nearest_nodes(
#                 network, X=row.geometry.x, Y=row.geometry.y
#             )
#             gdf.loc[idx, 'nearest_osm'] = [nearest_osm]  # 리스트로 저장
            
#         elif row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
#             # 격자 센트로이드들 생성
#             if row.geometry.geom_type == 'MultiPolygon':
#                 # MultiPolygon의 경우 가장 큰 폴리곤 선택
#                 polygon = max(row.geometry.geoms, key=lambda p: p.area)
#             else:
#                 polygon = row.geometry
            
#             grid_centroids = create_polygon_grid(polygon, grid_size)
            
#             # 각 격자 센트로이드의 nearest_osm 계산
#             nearest_nodes = []
#             for centroid in grid_centroids:
#                 nearest_node = ox.distance.nearest_nodes(
#                     network, X=centroid.x, Y=centroid.y
#                 )
#                 nearest_nodes.append(nearest_node)
            
#             gdf.loc[idx, 'nearest_osm'] = nearest_nodes
#         else:
#             print(f"Unsupported geometry type: {row.geometry.geom_type}")
#             continue
    
#     return gdf

# def nearest_osm_enhanced(network, gdf, grid_size=200):
#     gdf = gdf.copy()
#     gdf['nearest_osm'] = [[] for _ in range(len(gdf))]  # 빈 리스트 초기화
    
#     for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
#         geom = row.geometry
        
#         try:
#             if geom.geom_type == 'Point':
#                 nearest_node = ox.distance.nearest_nodes(
#                     network, X=geom.x, Y=geom.y
#                 )
#                 gdf.at[idx, 'nearest_osm'] = [nearest_node]
                
#             elif geom.geom_type in ['Polygon', 'MultiPolygon']:
#                 if geom.geom_type == 'MultiPolygon':
#                     geom = max(geom.geoms, key=lambda p: p.area)
#                 grid_centroids = create_polygon_grid(geom, grid_size)
#                 nearest_nodes = [
#                     ox.distance.nearest_nodes(network, X=pt.x, Y=pt.y)
#                     for pt in grid_centroids
#                 ]
#                 gdf.at[idx, 'nearest_osm'] = nearest_nodes
                
#         except Exception as e:
#             print(f"Error processing index {idx}: {str(e)}")
#             gdf.at[idx, 'nearest_osm'] = []  # 빈 리스트로 설정
            
#     return gdf
def nearest_osm_enhanced(network, gdf, grid_size=200):
    gdf = gdf.copy()
    
    # 1단계: object 타입 초기화 (강제)
    gdf['nearest_osm'] = gdf['nearest_osm'].astype('object')
    
    # 2단계: 빈 리스트로 초기화
    gdf['nearest_osm'] = [[] for _ in range(len(gdf))]
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        geom = row.geometry
        
        try:
            if geom.geom_type == 'Point':
                nearest_node = ox.distance.nearest_nodes(network, geom.x, geom.y)
                gdf.at[idx, 'nearest_osm'] = [nearest_node]  # 리스트 할당
                
            elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                if geom.geom_type == 'MultiPolygon':
                    geom = max(geom.geoms, key=lambda p: p.area)
                grid_centroids = create_polygon_grid(geom, grid_size)
                nearest_nodes = [
                    ox.distance.nearest_nodes(network, pt.x, pt.y)
                    for pt in grid_centroids
                ]
                gdf.at[idx, 'nearest_osm'] = nearest_nodes  # 리스트 할당
                
        except Exception as e:
            gdf.at[idx, 'nearest_osm'] = []  # 빈 리스트 유지
            
    # 3단계: 최종 타입 강제 지정
    gdf['nearest_osm'] = gdf['nearest_osm'].astype('object')
    
    return gdf





def step1_E2SFCA_enhanced(
    weights: Dict[Union[float, int], Union[float, int]],
    supply: gpd.GeoDataFrame,
    supply_attr: str,
    demand: gpd.GeoDataFrame,
    demand_attr: str,
    network: nx.MultiDiGraph,
    accessibility_threshold: float = 0.3  # 격자 중 접근 가능한 비율 임계값
):
    """
    비율을 고려한 격자 기반 E2SFCA 1단계 - 공급시설의 접근성 비율 계산
    """
    supply_ = supply.copy(deep=True)
    supply_['ratio'] = 0
    supply_['accessible_grid_ratio'] = 0  # 접근 가능한 격자 비율 추가
    
    for i in tqdm(range(supply_.shape[0])):
        supply_nearest_nodes = supply_.loc[i, 'nearest_osm']
        
        # 폴리곤의 경우 (격자 센트로이드들이 리스트로 저장됨)
        if isinstance(supply_nearest_nodes, list):
            accessible_grids = 0
            total_grids = len(supply_nearest_nodes)
            total_demand = 0
            
            for supply_node in supply_nearest_nodes:
                grid_demand = 0
                prev_nodes = set()
                
                # 각 격자 센트로이드에서 도달 가능한 수요 계산
                for time, weight in sorted(weights.items(), key=lambda x: x[0]):
                    temp_nodes = nx.single_source_dijkstra_path_length(
                        network, supply_node, cutoff=time, weight='time'
                    ).keys()
                    
                    current_nodes = set(temp_nodes) - prev_nodes
                    demand_sum = demand.loc[
                        demand['nearest_osm'].isin(current_nodes), demand_attr
                    ].sum() * weight
                    
                    grid_demand += demand_sum
                    prev_nodes.update(temp_nodes)
                
                # 해당 격자가 수요에 접근 가능한지 확인
                if grid_demand > 0:
                    accessible_grids += 1
                    total_demand += grid_demand
            
            # 접근 가능한 격자 비율 계산
            accessible_ratio = accessible_grids / total_grids if total_grids > 0 else 0
            supply_.loc[i, 'accessible_grid_ratio'] = accessible_ratio
            
            # 임계값 이상의 격자가 접근 가능한 경우에만 공급시설로 간주
            if accessible_ratio >= accessibility_threshold:
                supply_value = supply_.loc[i, supply_attr]
                # 접근 가능한 격자 비율을 가중치로 적용
                weighted_supply = supply_value * accessible_ratio
                step1_ratio = (weighted_supply / total_demand) * 100000 if total_demand > 0 else 0
                supply_.loc[i, 'ratio'] = step1_ratio
            else:
                supply_.loc[i, 'ratio'] = 0
                
        else:
            # Point의 경우 기존 로직 유지
            total_demand = 0
            prev_nodes = set()
            
            for time, weight in sorted(weights.items(), key=lambda x: x[0]):
                temp_nodes = nx.single_source_dijkstra_path_length(
                    network, supply_nearest_nodes, cutoff=time, weight='time'
                ).keys()
                
                current_nodes = set(temp_nodes) - prev_nodes
                demand_sum = demand.loc[
                    demand['nearest_osm'].isin(current_nodes), demand_attr
                ].sum() * weight
                
                total_demand += demand_sum
                prev_nodes.update(temp_nodes)
            
            supply_value = supply_.loc[i, supply_attr]
            step1_ratio = (supply_value / total_demand) * 100000 if total_demand > 0 else 0
            supply_.loc[i, 'ratio'] = step1_ratio
            supply_.loc[i, 'accessible_grid_ratio'] = 1.0  # Point는 100% 접근 가능
        
    return supply_

def step2_E2SFCA_enhanced(
    weights: Dict[Union[float, int], Union[float, int]],
    result_step1: pd.DataFrame,
    demand: pd.DataFrame,
    network: nx.Graph
) -> pd.DataFrame:
    """
    비율을 고려한 격자 기반 E2SFCA 2단계 - 수요지점의 접근성 지수 계산
    """
    demand_ = demand.copy(deep=True)
    demand_['access'] = 0
    
    for z in tqdm(range(demand_.shape[0]), desc="Processing demand points"):
        total_sum = 0
        prev_nodes = set()
        
        for time, weight in sorted(weights.items(), key=lambda x: x[0]):
            temp_nodes = nx.single_source_dijkstra_path_length(
                network,
                source=demand_.loc[z, 'nearest_osm'],
                cutoff=time,
                weight='time'
            ).keys()
            
            current_nodes = set(temp_nodes) - prev_nodes
            
            # 공급시설별 접근성 계산
            for idx, supply_row in result_step1.iterrows():
                supply_nearest_nodes = supply_row['nearest_osm']
                supply_ratio = supply_row['ratio']
                
                if supply_ratio == 0:  # 접근 불가능한 공급시설 제외
                    continue
                    
                if isinstance(supply_nearest_nodes, list):
                    # 격자 센트로이드 중 현재 노드 집합과 교차하는 것이 있는지 확인
                    accessible_grids = len([
                        node for node in supply_nearest_nodes 
                        if node in current_nodes
                    ])
                    
                    if accessible_grids > 0:
                        # 접근 가능한 격자 비율에 따라 접근성 조정
                        grid_accessibility_factor = accessible_grids / len(supply_nearest_nodes)
                        adjusted_ratio = supply_ratio * grid_accessibility_factor * weight
                        total_sum += adjusted_ratio
                else:
                    # Point 공급시설의 경우
                    if supply_nearest_nodes in current_nodes:
                        total_sum += supply_ratio * weight
            
            prev_nodes.update(temp_nodes)
        
        demand_.loc[z, 'access'] = total_sum

    return demand_

# def step1_E2SFCA_alternative(
#     weights: Dict[Union[float, int], Union[float, int]],
#     supply: gpd.GeoDataFrame,
#     supply_attr: str,
#     demand: gpd.GeoDataFrame,
#     demand_attr: str,
#     network: nx.MultiDiGraph
# ) -> gpd.GeoDataFrame:
#     """
#     대체 1단계: 모든 격자 캐치먼트의 합집합 기반 수요 계산
#     """
#     supply_ = supply.copy(deep=True)
#     supply_['ratio'] = 0

#     for i in tqdm(range(supply_.shape[0])):
#         supply_nearest_nodes = supply_.loc[i, 'nearest_osm']
        
#         if isinstance(supply_nearest_nodes, list):
#             # 모든 격자 캐치먼트의 합집합 계산
#             all_demand_nodes = set()
#             for supply_node in supply_nearest_nodes:
#                 prev_nodes = set()
#                 total_grid_demand = 0
                
#                 # 시간 계층별 누적 노드 수집
#                 for time, _ in sorted(weights.items(), key=lambda x: x[0]):
#                     temp_nodes = nx.single_source_dijkstra_path_length(
#                         network, supply_node, cutoff=time, weight='time'
#                     ).keys()
#                     current_nodes = set(temp_nodes) - prev_nodes
#                     all_demand_nodes.update(current_nodes)
#                     prev_nodes.update(temp_nodes)
            
#             # 고유 수요 노드 기반 총 수요량 계산
#             total_demand = demand.loc[
#                 demand['nearest_osm'].isin(all_demand_nodes), demand_attr
#             ].sum()
            
#             # 공급 비율 계산
#             if total_demand > 0:
#                 supply_value = supply_.loc[i, supply_attr]
#                 supply_.loc[i, 'ratio'] = (supply_value / total_demand) * 100000
#             else:
#                 supply_.loc[i, 'ratio'] = 0
                
#         else:
#             # 점형 시설은 기존 로직 유지
#             total_demand = 0
#             prev_nodes = set()
            
#             for time, weight in sorted(weights.items(), key=lambda x: x[0]):
#                 temp_nodes = nx.single_source_dijkstra_path_length(
#                     network, supply_nearest_nodes, cutoff=time, weight='time'
#                 ).keys()
                
#                 current_nodes = set(temp_nodes) - prev_nodes
#                 demand_sum = demand.loc[
#                     demand['nearest_osm'].isin(current_nodes), demand_attr
#                 ].sum() * weight
                
#                 total_demand += demand_sum
#                 prev_nodes.update(temp_nodes)
            
#             if total_demand > 0:
#                 supply_value = supply_.loc[i, supply_attr]
#                 supply_.loc[i, 'ratio'] = (supply_value / total_demand) * 100000
#             else:
#                 supply_.loc[i, 'ratio'] = 0

#     return supply_

def step1_E2SFCA_alternative(
    weights: Dict[Union[float, int], Union[float, int]],
    supply: gpd.GeoDataFrame,
    supply_attr: str,
    demand: gpd.GeoDataFrame,
    demand_attr: str,
    network: nx.MultiDiGraph
) -> gpd.GeoDataFrame:
    """
    개선된 1단계: 공원(parent_id) 단위 캐치먼트 통합 계산
    """
    supply_ = supply.copy(deep=True)
    supply_['ratio'] = 0

    # 공원별 그룹화
    for parent_id, group in tqdm(supply_.groupby('parent_id'), desc="Processing parks"):
        all_demand_nodes = set()
        
        # 모든 격자 노드의 캐치먼트 통합
        for idx, row in group.iterrows():
            nodes = row['nearest_osm']
            if isinstance(nodes, list):
                for node in nodes:
                    prev = set()
                    for time, _ in sorted(weights.items()):
                        temp = nx.single_source_dijkstra_path_length(
                            network, node, cutoff=time, weight='time'
                        ).keys()
                        all_demand_nodes.update(set(temp) - prev)
                        prev.update(temp)
            else:
                prev = set()
                for time, _ in sorted(weights.items()):
                    temp = nx.single_source_dijkstra_path_length(
                        network, nodes, cutoff=time, weight='time'
                    ).keys()
                    all_demand_nodes.update(set(temp) - prev)
                    prev.update(temp)

        # 고유 수요 계산
        total_demand = demand.loc[
            demand['nearest_osm'].isin(all_demand_nodes), demand_attr
        ].sum()

        # 비율 계산 및 할당
        ratio = (group[supply_attr].iloc[0] / total_demand * 100000) if total_demand > 0 else 0
        supply_.loc[group.index, 'ratio'] = ratio

    return supply_


# def step2_E2SFCA_alternative(
#     weights: Dict[Union[float, int], Union[float, int]],
#     result_step1: pd.DataFrame,
#     demand: pd.DataFrame,
#     network: nx.Graph
# ) -> pd.DataFrame:
#     """
#     대체 2단계: 단일 격자 포함 시 전체 공급량 반영
#     """
#     demand_ = demand.copy(deep=True)
#     demand_['access'] = 0
    
#     for z in tqdm(range(demand_.shape[0]), desc="Processing demand points"):
#         total_sum = 0
#         prev_nodes = set()
        
#         for time, weight in sorted(weights.items(), key=lambda x: x[0]):
#             temp_nodes = nx.single_source_dijkstra_path_length(
#                 network,
#                 source=demand_.loc[z, 'nearest_osm'],
#                 cutoff=time,
#                 weight='time'
#             ).keys()
            
#             current_nodes = set(temp_nodes) - prev_nodes
            
#             # 공급시설 접근성 계산
#             for idx, supply_row in result_step1.iterrows():
#                 supply_nearest_nodes = supply_row['nearest_osm']
#                 supply_ratio = supply_row['ratio']
                
#                 if supply_ratio == 0:
#                     continue
                    
#                 # 폴리곤 시설: 단일 격자 포함 여부 확인
#                 if isinstance(supply_nearest_nodes, list):
#                     if any(node in current_nodes for node in supply_nearest_nodes):
#                         total_sum += supply_ratio * weight
#                 # 점형 시설: 기존 로직
#                 else:
#                     if supply_nearest_nodes in current_nodes:
#                         total_sum += supply_ratio * weight
            
#             prev_nodes.update(temp_nodes)
        
#         demand_.loc[z, 'access'] = total_sum

#     return demand_

def step2_E2SFCA_alternative(
    weights: Dict[Union[float, int], Union[float, int]],
    result_step1: pd.DataFrame,
    demand: pd.DataFrame,
    network: nx.Graph
) -> pd.DataFrame:
    """
    개선된 2단계: 공원 단위 중복 제거 계산
    """
    # 공원별 데이터 사전 생성
    park_data = result_step1.groupby('parent_id').agg({
        'ratio': 'first',
        'nearest_osm': lambda x: list(set([n for lst in x for n in lst]))
    }).to_dict('index')

    demand_ = demand.copy()
    demand_['access'] = 0

    for z in tqdm(range(demand_.shape[0]), desc="Demand points"):
        total = 0
        prev = set()
        
        for time, weight in sorted(weights.items()):
            current = set(nx.single_source_dijkstra_path_length(
                network, demand_.loc[z, 'nearest_osm'], cutoff=time, weight='time'
            ).keys()) - prev
            
            # 공원별 접근성 확인
            for park_id, data in park_data.items():
                if any(node in current for node in data['nearest_osm']):
                    total += data['ratio'] * weight
            
            prev.update(current)
        
        demand_.loc[z, 'access'] = total

    return demand_


################## 성능 최적화 방안 ##########################

# 병렬 처리 적용 예시 (공급측만)

from multiprocessing import Pool

def process_supply(supply_data):
    return nearest_osm_enhanced(network, supply_data)

################## ADB 작업용 코드 ##########################
def process_rwi_to_grid(
    boundary_shp_path, 
    input_rwi_path, 
    output_path, 
    utm_epsg_uzb=32641
):
    '''우즈백 상대적 부 지수 산출 코드 파이프라인'''
    # 1. 격자 데이터 로드
    grid_gdf = gpd.read_file(boundary_shp_path).to_crs(epsg=utm_epsg_uzb)
    grid_gdf["grid_id"] = grid_gdf.index

    # 2. RWI 포인트 데이터 로드 및 변환
    rwi_df = pd.read_csv(input_rwi_path)
    rwi_gdf = gpd.GeoDataFrame(
        rwi_df,
        geometry=gpd.points_from_xy(rwi_df.longitude, rwi_df.latitude),
        crs="EPSG:4326"
    ).to_crs(epsg=utm_epsg_uzb)

    # 3. 공간 조인 및 격자별 평균 계산
    joined_rwi = gpd.sjoin(rwi_gdf, grid_gdf, how="left", predicate="intersects")
    rwi_grid = joined_rwi.groupby("index_right")["rwi"].mean().reset_index()

    # 4. 결과 병합 및 결측값 처리
    result = grid_gdf.merge(
        rwi_grid,
        left_on="grid_id",
        right_on="index_right",
        how="left"
    )
    result["rwi"] = result["rwi"].fillna(0)

    # 5. 저장
    result.to_file(output_path, driver="GeoJSON")
    print(f"✅ 처리 완료: {output_path}")
