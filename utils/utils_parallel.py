# 별도 모듈 작성 (utils_parallel.py)
import utils

def process_chunk(chunk, network, grid_size):
    return utils.nearest_osm_enhanced(
        network=network, 
        gdf=chunk, 
        grid_size=grid_size
    )
