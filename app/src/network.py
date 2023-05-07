import pandas as pd
import numpy as np
import polars as pl
import seaborn as sns
import plotly as px
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import sranodec as anom

# cpu = pd.read_pickle('../../data/cpu.pkl') # to read cpu pickle data
# mem = pd.read_pickle('../../data/memory.pkl') # to read memory pickle data
net = pd.read_pickle('../../data/network.pkl') 

# Now we change the dataframes to polars for faster processing
net = pl.from_pandas(net)

# Change from string to datetime in the same format of time.
net = net.with_columns(pl.col('timestamp').apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')))

net2 = (
    net
    .lazy()
    .with_columns(
        ((pl.col('network.out.bytes') / (pl.col('network.out.bytes')+pl.col('network.in.bytes'))*100).alias('out.pct')),
        ((pl.col('network.in.bytes') / (pl.col('network.out.bytes')+pl.col('network.in.bytes'))*100).alias('in.pct'))
    )
    .collect()
)

net2 = net2.drop(columns=['network.out.bytes', 'network.in.bytes'])

out2 = net2.to_pandas()

def spectral_anomaly_plot(df, amp_window_size=24, series_window_size=24, score_window_size=100, 
                          network='in.pct', node = None, timerange = None) -> None:
    """
    This function plots the spectral anomaly plot for a given network and node.
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the network data.
    amp_window_size : int
        The size of the window for the amplitude.
    series_window_size : int
        The size of the window for the series.
    score_window_size : int
        The size of the window for the score.
    network : str
        The network to be plotted.
    node : str
        The node to be plotted.
    timerange : tuple
        The timerange to be plotted.
    Returns
    -------
    None
    """
    # Creates the silency object which stores our three window sizes. This variables are later used to calculate each
    # of the three components of the silency algorithm and help us do spectral analysis.

    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)

    # We create a test signal which is the signal we want to test for anomalies. In this case we are testing the
    # network in percentage for the node 'node1'. We can also test for all nodes by setting node = None.
    if node:
        test_signal = df[df['node'] == node][network]
    else:
        test_signal = df[network]

    # We can also set a timerange to test for anomalies. This is useful when we want to test for anomalies in a
    # specific time period.
    if timerange:
        test_signal = test_signal[(test_signal.index > timerange[0]) & (test_signal.index < timerange[1])]
    

    # We generate the anomaly score for the test signal. The anomaly score is a measure of how anomalous each point
    # in the test signal is. The higher the score, the more anomalous the point is.
    # The score is based on the following function (magnitude - average_filter) / average_filter
    # where magnitude is the magnitude of the signal, and average_filter is calculated the following way
    # cumulative sum of the values before the point of our data including those up to the kernel size
    #  minus the cumulative values after the point in degree of the kernel and then all divided by the kernel

    score = spec.generate_anomaly_score(test_signal)
    plt.figure(figsize=(20, 6))
    plt.plot(test_signal, alpha=0.5, label="observation")
    if node == None:
        node = 'all' # changed so that title doesn't say none
    plt.title(f'{node} anomalies for {network}')
    index_changes = np.where(score > np.percentile(score, 99))[0] # percentile can be changed, 99% is a good start.
    plt.scatter(test_signal.iloc[index_changes].index, y = test_signal.iloc[index_changes], c='green', label="change point")