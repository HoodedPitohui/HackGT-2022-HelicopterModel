import matplotlib
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

save_path = os.path.join("data")

def parse_data(given_path):
    file_name = 'metars_cache_1.csv'
    metars_path = os.path.join(given_path, file_name)
    metars_data = pd.read_csv(metars_path)
    return metars_data

def main():
    given_data = parse_data(save_path)
    plt.scatter(given_data['latitude'], given_data['longitude'])
    plt.show()
    plt.scatter(given_data['latitude'], given_data['altim_in_hg'])


if __name__ == "__main__":
    main()