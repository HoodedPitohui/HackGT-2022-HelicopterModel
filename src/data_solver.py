import matplotlib
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from scipy.interpolate import interp1d
import heapq
import pickle
import urllib.request
import csv
from scipy import stats


pd.options.mode.chained_assignment = None 

#default save paths
save_path = os.path.join("HackGT-2022-HelicopterModel")
src_path = os.path.join("src")
map_path = os.path.join("maps")
save_image_path = os.path.join("saved_figures")

class Node:
    #Used for A-star algorithm
    #credit: https://gist.github.com/ryancollingwood/32446307e976a11a1185a5394d6657bc
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path



def parse_data(given_path):
    # metars_path = os.path.join(given_path, 'metars.cache.csv')
    data = []
    with open('metars.cache.csv', 'r') as datafile:
        reader = csv.reader(datafile, delimiter=',')
        for row in reader:
            data.append(row)
    
    # Get rid of the first few lines which are pointless
    data = data[5:]

    with open('metars.cache.csv', 'w') as processedFile:
        writer = csv.writer(processedFile)
        for row in data:
            writer.writerow(row)

    metars_data = pd.read_csv('metars.cache.csv')

    metars_data.sort_values('latitude')
    #sort by latitude
    return metars_data

def plot_map(df: pd.DataFrame, title: str, save_name: str, region: str=None, xtitle: str = None, ytitle: str = None,
    color_var: str = None, cmap: str = None, elim_outliers: bool = False) -> None:
    """
    Takes in a subset of the data, applies relevant titles, and provides the option of visualizing variations in a 
    particular quantity using a colorbar

    ---

    Notes about certain inputs:
    region - a string that's supposed to be either "world" or "usa"
    """
    # If no region is specified, default to the world
    if not region:
        region = 'world'
    
    # Get the appropriate map file. This segment is inspired by https://towardsdatascience.com/easiest-way-to-plot-on-a-world-map-with-pandas-and-geopandas-325f6024949f
    if region == 'usa':
        # For ease of use, we're excluding Hawaii and Alaska
        map = gpd.read_file(os.path.join(map_path, 's_22mr22.shp'))
        map = map[map.STATE != 'AK']
        map = map[map.STATE != 'HI']
    else:
        map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(12, 6))
    map.plot(color="lightgrey", ax=ax)

    # If color_var was specified, then reassign itself to the series corresponding to df[color_var]
    if color_var:
        # Specifically clean based on this row to remove NaN values
        df = df.dropna(axis=0, subset=[color_var])
        df = df.reset_index(drop=True)

        # Remove outliers that are more than 3 standard dev's away from the mean
        # Based on https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe
        if elim_outliers:
            # This is off by default
            df = df[(np.abs(stats.zscore(df[color_var])) < 3)]

        color_var_data = df[color_var]

    plt.scatter(df['longitude'], df['latitude'], s=1, c=color_var_data, cmap=cmap)
    if color_var:
        plt.colorbar(label=color_var)

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    # If the region was the US, apply axis limits
    if region == 'usa':
        plt.xlim([-125, -65])
        plt.ylim([24, 50])

    plt.savefig(os.path.join(save_image_path, save_name))

def filter_usa(df: pd.DataFrame) -> pd.DataFrame:
    """Takes in world data and crudely restricts to just lat/long values corresponding to the lower 48"""

    # We'll be using latitude bounds of [24, 49] and longitude bounds of [-125, -67] based on https://www.findlatitudeandlongitude.com/l/Lower+48/4315442/
    filtered = df.loc[(df['latitude'] >= 24) & (df['latitude'] <= 49) & (df['longitude'] >= -125) & (df['longitude'] <= -67)]

    # Also, because 'Murica, let's change the units from C to F
    filtered['temp_f'] = filtered['temp_c'].apply(lambda x: (x * (9.0 / 5.0) + 32))
    filtered['dewpoint_f'] = filtered['dewpoint_c'].apply(lambda x: (x * (9.0 / 5.0) + 32))

    return filtered


def visualize_conditions(data):
    # Do some basic cleaning to get rid of rows that are missing lat/long, since that's the bare minimum we need
    cleaned_data = data.dropna(axis=0, subset=['latitude', 'longitude'])
    cleaned_data = cleaned_data.reset_index(drop=True)
    
    usa_data = filter_usa(cleaned_data)

    # Now store all of the output plots
    plot_map(usa_data, 'Temperature Across the USA in Fahrenheit', 'usa_temp.png', region='usa', color_var='temp_f', cmap='autumn_r')
    plot_map(usa_data, 'Dewpoint Temperature Across the USA in Fahrenheit', 'usa_dewpoint.png', region='usa', color_var='dewpoint_f', cmap='autumn_r')
    plot_map(usa_data, 'Wind Speed in Knots Across the USA', 'usa_wind_speed.png', region='usa', color_var='wind_speed_kt', cmap='autumn_r')
    plot_map(usa_data, 'Visibility Statute in Miles Across the USA', 'usa_visibility.png', region='usa', color_var='visibility_statute_mi', cmap='autumn_r', elim_outliers=True)

def clean_null_rows(given_data):
    #get rid of null rows
    cleaned_data = given_data.copy()
    cleaned_data.dropna(subset = ['temp_c'], inplace = True)
    cleaned_data.dropna(subset = ['dewpoint_c'], inplace = True)
    cleaned_data.dropna(subset = ['visibility_statute_mi'], inplace = True)
    return cleaned_data

def get_na_data(given_data):
    #get data in NA
    na_data = given_data[given_data.latitude > 31]
    na_data = na_data[na_data.latitude < 51]
    na_data = na_data[na_data.longitude > -180]
    na_data = na_data[na_data.longitude < -50]
    return na_data

def geoplot_data_world(given_data, save_name):
    #use geoplot for preliminary mapping

    geometry = [Point(xy) for xy in zip(given_data['longitude'], given_data['latitude'])]
    gdf = GeoDataFrame(given_data, geometry=geometry)   
    #this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=0.5)
    plt.savefig(os.path.join(save_image_path, save_name))



def plot_user_points(given_data, given_lats, given_lons, save_name): 
    #plot a user given dataset and save
    plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5)
    plt.scatter(given_lons, given_lats, s = 20)
    plt.savefig(os.path.join(save_image_path, save_name))
    plt.close()
    #plt.show()

def find_relevant_points(given_data, user_lats, user_lons):
    #find points within the boundaries of what the user specified as their latitude annd longitude coordinates
    max_lat = max(user_lats)
    min_lat = min(user_lats)
    max_lon = max(user_lons)
    min_lon = min(user_lons)

    relevant_data = given_data[given_data.latitude < max_lat]
    relevant_data = relevant_data[relevant_data.latitude > min_lat]
    relevant_data = relevant_data[relevant_data.longitude < max_lon]
    relevant_data = relevant_data[relevant_data.longitude > min_lon]
    return relevant_data

def generate_risk_assessments(given_data, wind_speed, temp_c, dewpoint_c):
    # Glorious risk model
    model_path = os.path.join(src_path, 'model.pickle')
    model = pickle.load(open(model_path, 'rb'))

    inputs = given_data['visibility_statute_mi']
    
    risk_array = np.empty((len(given_data), 1))
    for i in range(0, len(given_data)):
        # Get prediction for this data entry
        prediction = model.predict(pd.DataFrame([float(inputs.iloc[i])], columns=['visibility_statute_mi']))

        # Assign risk based on prediction
        if prediction == 'SPECI':
            risk_array[i] = 300
        else:
            risk_array[i] = 0
        
    return risk_array

def interpolate_metrics(given_data, point_density, risk_array): #given_data should be sorted ascending in altitude
    #example point_density = 0.01
    lat_range = max(given_data['latitude']) - min(given_data['latitude'])
    lat_length = round(lat_range / point_density)
    long_range = max(given_data['longitude']) - min(given_data['longitude'])
    long_length = round(long_range / point_density)
    interp_vec = np.empty((lat_length * long_length, 3))
    for i in range(0, lat_length):
        for j in range(0, long_length):
            interp_vec[i * long_length + j, 0] = min(given_data['latitude']) + point_density * i
            interp_vec[i * long_length + j, 1] = min(given_data['longitude']) + point_density * j
            neighbor_array, distance_array = find_nearest_neighbors(np.asarray(given_data['latitude']), 
                                                                    np.asarray(given_data['longitude']), 
                                                                    interp_vec[i * long_length + j, 0], 
                                                                    interp_vec[i * long_length + j, 1], 5)
        
            interp_vec[i * long_length + j, 2] = calculate_weighted_risk(neighbor_array, distance_array, risk_array)

    return interp_vec, long_length, lat_length

def calculate_weighted_risk(neighbor_array, distance_array, risk_array):
    weighted_risk = 0
    for i in range(0, len(neighbor_array)):
        test = neighbor_array[i]
        weighted_risk = weighted_risk + risk_array[int(neighbor_array[i])] / (distance_array[0, i] ** 1.25 + 1)
    return weighted_risk
    
def find_nearest_neighbors(given_lats, given_lons, lat, lon, num_neighbors):

    distances = np.empty((1, len(given_lats)))
    for i in range(0, len(given_lats)):
        distances[0, i] = ((lat - given_lats[i]) **2 + (lon - given_lons[i]) **2) **(1/2)
    
    sorted_distances = np.sort(distances)
    neighbor_array = np.empty((num_neighbors, 1))
    for i in range(0, len(neighbor_array)):
        f_index = np.where(distances == sorted_distances[0, i])
        if (len(f_index[1]) > 1):
            neighbor_array[i] = f_index[0][0]
        else:
            neighbor_array[i] = f_index[1]

    return neighbor_array, sorted_distances[0: 5]

def convert_weighted_graph(penalty_graph, long_length, lat_length):
    construction_graph = np.empty((lat_length, long_length))
    for i in range(0, lat_length):
        for j in range(0, long_length):
            construction_graph[i, j] = penalty_graph[i * lat_length + j, 2]
    return construction_graph

def calc_diag_penalty(given_penalties, long_length, lat_length):
    #get a penalty for the simplest action
    construction_graph_costs = convert_weighted_graph(given_penalties, long_length, lat_length)
    starting_node = np.asarray([0, 0])
    ending_node = np.asarray([lat_length - 1, long_length - 1])

    #calculate the diagonal-safety-cost
    diag_safety_cost = 0
    final_pos_counter = 0
    if (long_length > lat_length):
        diag_path = np.empty([long_length, 2])
        for i in range(0, lat_length):
            diag_safety_cost = diag_safety_cost + construction_graph_costs[i, i]
            diag_path[i] = [i, i]
            final_pos_counter += 1
        for  i in range(final_pos_counter, long_length):
            diag_safety_cost = diag_safety_cost + construction_graph_costs[final_pos_counter - 1, i]
            diag_path[i] = [i, final_pos_counter]
    else:
        diag_path = np.empty([lat_length, 2])
        for i in range(0, long_length):
            diag_safety_cost = diag_safety_cost + construction_graph_costs[i, i]
            diag_path[i] = [i, i]
            final_pos_counter += 1
        for i in range(final_pos_counter, lat_length):
            diag_safety_cost = diag_safety_cost + construction_graph_costs[i, final_pos_counter]
            diag_path[i] = [final_pos_counter, i]

    return diag_safety_cost, diag_path, construction_graph_costs

def a_star(penalty_graph, long_length, lat_length):
    #a-star for best pathing
    #credit: https://gist.github.com/ryancollingwood/32446307e976a11a1185a5394d6657bc
    #assumption, bottom left = start, end is top right
    start = (0, 0)
    end = (lat_length - 1, long_length - 1)

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    outer_iterations = 0
    max_iterations = (len(penalty_graph[0]) * len(penalty_graph))
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    while len(open_list) > 0:
        outer_iterations += 1
        if outer_iterations > max_iterations:
            return return_path(current_node)

        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        if current_node == end_node:
            return return_path(current_node)
        
        children = []

        for new_position in adjacent_squares:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(penalty_graph) - 1) or node_position[0] < 0 or node_position[1] > (len(penalty_graph[len(penalty_graph)-1]) -1) or node_position[1] < 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)
        
        for child in children:
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue
            
            child.g = current_node.g + penalty_graph[child.position[0], child.position[1]]
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            heapq.heappush(open_list, child)
    

def get_path_cost(penalty_graph, best_path):

    #get path cost from given actions
    cost = 0
    for i in range(0, len(best_path)):
        cost = cost + penalty_graph[best_path[i]]
    return cost

def get_lat_lons_from_path(best_path, lat_lon_data, lat_len, long_len):
    #generate latitudes and longitudes for the solution to A*
    lat_lon_table = np.empty((len(best_path), 2))
    for i in range(0, len(best_path)):
        temp = np.asarray(best_path[i])
        row_num = temp[0] * long_len + temp[1]
        lat_lon_table[i, 0] = lat_lon_data[row_num, 0]
        lat_lon_table[i, 1] = lat_lon_data[row_num, 1]
    
    return lat_lon_table

def plot_user_final_path(given_data, path_data, given_lats, given_lons, save_name):
    plt.scatter(given_data['longitude'], given_data['latitude'], s = 0.5)
    plt.scatter(given_lons, given_lats, s = 20)
    plt.plot(path_data[:, 1], path_data[:, 0])
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Helicopter from Atlanta, GA to Durham, NC')
    plt.savefig(os.path.join(save_image_path, save_name))
    plt.close()
    x = 8

def main():
    urllib.request.urlretrieve("https://www.aviationweather.gov/adds/dataserver_current/current/metars.cache.csv", "metars.cache.csv")
    given_data = parse_data(save_path)
    #plot_data_world(given_data, 'initial_lat_lon.png')

    #clean out null rows
    cleaned_data = clean_null_rows(given_data)
    #plot_data_world(cleaned_data, 'cleaned_lat_lon.png')
    na_data = get_na_data(cleaned_data)

    #pick two points and find useful datapoints from there
    lat1, lon1 = 33.7490, -84.3880
    lat2, lon2 = 35.9940, -78.8986

    given_lats = [lat1, lat2]
    given_lons = [lon1, lon2]
    plot_user_points(na_data, given_lats, given_lons, 'atlanta-durham_general.png')
    relevant_user_data = find_relevant_points(na_data, given_lats, given_lons)
    plot_user_points(relevant_user_data, given_lats, given_lons, 'atlanta-durham_narrowed.png')
    risk_array = generate_risk_assessments(relevant_user_data, np.asarray(relevant_user_data['wind_speed_kt']), 
                                            np.asarray(relevant_user_data['temp_c']), 
                                            np.asarray(relevant_user_data['dewpoint_c']))
    weighted_risk_matrix, long_length, lat_length = interpolate_metrics(relevant_user_data, 0.1, risk_array)
    diag_path_cost, diag_path, penalty_graph = calc_diag_penalty(weighted_risk_matrix, long_length, lat_length)
    best_path= a_star(penalty_graph, long_length, lat_length)
    best_path_cost = get_path_cost(penalty_graph, best_path)
    lat_lon_path = get_lat_lons_from_path(best_path, weighted_risk_matrix, lat_length, long_length)
    np.resize(lat_lon_path, (len(lat_lon_path) + 1, 2))
    lat_lon_path[len(lat_lon_path) - 1] = [lat2, lon2]
    plot_user_final_path(relevant_user_data, lat_lon_path, given_lats, given_lons, 'atlanta_durham_path.png')

    # Use the given data to generate visualizations over the lower 48
    visualize_conditions(given_data)

if __name__ == "__main__":
    main()