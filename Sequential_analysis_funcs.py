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

def trend(time, slope=0):
    '''
    Generate a linear trend in the time series data
    :param time: Time series data
    :param slope: Slope of the trend
    :return: Trend data

    '''
    return slope * time

def seasonal_pattern(season_time):
    '''
    Generate an arbitrary seasonal pattern
    Args:
        season_time (array of float): measurements per time step
    Returns:
        revised measurement values based on the pattern 

    '''
    data_pattern = np.where(season_time < 0.5,
                            np.cos(season_time * 2* np.pi),
                            1/ np.exp(3 * season_time))
    return data_pattern

def seasonality(time, period, amplitude=1, phase=0):
    '''
    Generate a time series with repeated seasonal pattern
    Args:
        time (array of float): time steps
        period (float): period of the seasonal pattern
        amplitude (float): amplitude of the seasonal pattern
        phase (float): phase of the seasonal pattern
    Returns:
        time series with seasonal pattern

    '''
    # Values per period
    season_time = ( ( time + phase) % period )/ period # This makes the data seasonal with values between 0 and 1
    
    # Generate the pattern
    data_pattern = amplitude* seasonal_pattern(season_time) # This gives the desired pattern to our seasonal data

    return data_pattern

def sum(a,b):
    return a+b

def noise(time, noise_level=1, seed=None):
    '''
    Generate noise to add in a time series data
    Args:
        time (array of float): time steps
        noise_level (float): noise level
        seed (int): seed for random number generator
    Returns:
        time series of noise
    
    '''

    # Initialise the random number generator
    rnd = np.random.RandomState(seed)

    # Generate a random number for each time step and scale by the noise_level
    noise_series = rnd.randn(len(time)) * noise_level

    return noise_series

def autocorrelation(time, amplitude, seed=None):
    '''
    Generate autocorrelated data.
    Args:
        time (array of float): time steps
        amplitude (float): amplitude of the autocorrelation
        seed (int): seed for random number generator
    Returns:
        autocorrelated data
    '''

    # Initialise the random number generator
    rnd = np.random.RandomState(seed)

    # Define scaling factors
    phi1 = 0.5
    phi2 = -0.1

    # Generate a series with noise
    ar = rnd.randn(len(time) + 50)
    # extra 50 points because we want to correlate with t-50 and t-30.

    # Make the first 50 points constants
    ar[:50] = 200

    # Autocorrelate element 50 onwards with t-50 and t-30
    for step in range(50, len(time +50)):
        ar[step] += phi1* ar[step-50] + phi2* ar[step-30]
    
    # Return the series minus the first 50 points
    return amplitude* ar[50:]


def moving_average(series, window):
    '''
    Compute the moving average of a time series
    Args:
        series (array of float): time series data
        window (int): window size
    Returns:
        moving average of the time series data

    '''

    # Initialise the moving average list
    mov_avg = []

    # Loop through the series
    for time in range(len(series) - window):
        mov_avg.append(series[time: time+ window].mean())
    
    # Convert to a numpy array and return
    return np.array(mov_avg)