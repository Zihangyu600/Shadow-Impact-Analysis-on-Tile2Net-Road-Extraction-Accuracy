import geopandas as gpd
import matplotlib.pyplot as plt
import math
import os


def tile_to_coordinates(zoom, x, y):
    """将瓦片坐标转换为经纬度边界"""
    n = 2.0 ** zoom

    # 经度计算
    lon_west = x / n * 360.0 - 180.0
    lon_east = (x + 1) / n * 360.0 - 180.0
    lon_east = 2*lon_east-1*lon_west

    # 纬度计算
    lat_north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_south = 2*lat_south-1*lat_north
    #print(lat_north, lat_south)
    return (lon_west, lat_south, lon_east, lat_north) # [minx, miny, maxx, maxy]


def geojson_to_png_with_bounds(geojson_path, png_path, bounds):
    """
    将GeoJSON转换为PNG，并指定坐标范围

    Parameters:
    - geojson_path: GeoJSON文件路径
    - png_path: 输出PNG路径
    - bounds: 坐标范围 [minx, miny, maxx, maxy]
    - width, height: 输出图片尺寸
    """
    # 读取GeoJSON文件
    gdf = gpd.read_file(geojson_path)

    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制道路
    gdf.plot(ax=ax, color='#377E22', linewidth=1)

    # 设置指定的坐标范围
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # 设置宽高比
    ax.set_aspect('equal')

    # 隐藏坐标轴
    ax.set_axis_off()

    # 保存为PNG
    plt.savefig(png_path, dpi=400, bbox_inches='tight', pad_inches=0,
                facecolor='black', transparent=False)
    plt.close()
    print(f"save to: {png_path}")

#wright your own input depend on the pictures in tiles
x1 = 154359 #the x coord from first picture in tiles
x2 = 154368
#x2 = 154406 #the x coord from last picture in tiles
y1 = 197062 #the y coord from first picture in tiles
y2 = 197070
#y2 = 197125 #the y coord from last picture in tiles

for i in range(x1,x2):
    for j in range(y1,y2):
        x_coord = i
        y_coord = j
        zoom = 19
        x, y = x_coord,y_coord

        bounds = tile_to_coordinates(zoom, x, y)
        print(bounds)
        print(f"\n瓦片地理坐标范围:")
        print(f"经度范围: {bounds[0]:.6f} 到 {bounds[2]:.6f}")
        print(f"纬度范围: {bounds[1]:.6f} 到 {bounds[3]:.6f}")

        # 生成道路掩膜图片
        geojson_path = r'C:\Users\yzh03\PycharmProjects\PythonProject1\data\raw\Roadbed_2022_-6606727984785838060.geojson' # 你的GeoJSON文件路径
        output_png = f"D:/data/outputdir/new york/segmentation/new york/256_19_8/19/{x}_{y}.png"  # 输出文件名

        geojson_to_png_with_bounds(geojson_path, output_png, bounds)

from tile2net import Raster
import geopandas as gpd


location_ny = '40.4978740560533,-74.2550385911449,40.9151395390951,-73.6996327792974'
location_ny_small = '40.714,-73.980,40.744,-74.010'

raster = Raster(
    location=location_ny_small,
    zoom = 19,
    name='new york',
    input_dir='<D:/data/outputdir/new york/segmentation/new york/256_19_8/z/x/y.png>',
    output_dir=r"D:\data\outputdir\oldyork",
    dump_percent=100,

)
raster

raster.generate(4)
