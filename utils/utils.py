import geopandas as gpd
import pandas as pd
import osmnx as ox
import time
import numpy as np
from tqdm import tqdm, trange
from shapely.geometry import Point, MultiPoint
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