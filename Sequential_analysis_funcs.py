# A bunch of helper functions for sequential analysis, mostly for analysis on time series data.
# Author: Prashant Saxena
# Date: 7 March 2025

import numpy as np
import matplotlib.pyplot as plt


# Plotting a time series
def plot_series(time, series, format="-", start=0, end=None, label=None):
    '''
    Visualize the time series data
    :param time: Time series data
    :param series: Series data
    :param format: Format of the plot
    :param start: Start index of the plot
    :param end: End index of the plot
    :param label: Label for the plot
    :return: None


    '''

    plt.figure(figsize=(10, 6))

    if type(series) is tuple:
        for i in range(len(series)):
            plt.plot(time[start:end], series[i][start:end], format, label=label[i])
    else:
        plt.plot(time[start:end], series[start:end], format, label=label)
    
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()


