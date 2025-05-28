# 별도 모듈 작성 (utils_parallel.py)
import utils

def process_chunk(chunk, network, grid_size):
    return utils.nearest_osm_enhanced(
        network=network, 
        gdf=chunk, 
        grid_size=grid_size
    )

# utils_parallel.py
import osmnx as ox  # 필수 임포트 추가
import geopandas as gpd
import pandas as pd
import time
import numpy as np
from tqdm import tqdm, trange
from shapely.geometry import Point, MultiPoint
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import pearsonr
from matplotlib_scalebar.scalebar import ScaleBar
#from shapely.ops import cascaded_union, unary_union
import utils
import warnings
warnings.filterwarnings("ignore")



# def process_chunk(args):
#     network, chunk = args
#     chunk = chunk.copy()
    
#     # nearest_osm 컬럼 초기화
#     if 'nearest_osm' not in chunk.columns:
#         chunk['nearest_osm'] = None
    
#     for idx, row in chunk.iterrows():
#         point = row.geometry
#         try:
#             # OSMnx 함수 사용
#             nearest_node = ox.distance.nearest_nodes(network, point.x, point.y)
#             chunk.at[idx, 'nearest_osm'] = int(nearest_node)  # 정수형 변환
#         except Exception as e:
#             print(f"Error at index {idx}: {e}")
#             chunk.at[idx, 'nearest_osm'] = None
    
#     return chunk
