import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.ensemble import IsolationForest
import altair as alt
from sklearn.neighbors import KernelDensity
import sranodec as anom
import numpy as np

st.set_page_config(
    page_title="Network Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('## The data')
st.markdown('The data for this particular system is divided into node, which is a commonality among all of the three systems, incoming traffic in bytes or **in**, and outgoing traffic in bytes or **out**')
st.markdown('For the data we did a resampling of 1 minute. We saw that updates came once every 20 seconds, so we decided to compress the data to make it easier to read.')
st.markdown('To standarize data among all nodes we decided to change in and out bytes to a percentage. Meaning we have the percent of in bytes agains all bytes at that time and the same for out bytes')

st.title("Network Monitoring")

with st.spinner("Cargando, espera..."):
    network = pd.read_pickle("../data/network.pkl")
    network["timestamp"] = network["timestamp"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    network = network.resample('1min', on="timestamp").agg("first").ffill()
    network['in_pct'] = (network['network.in.bytes'] / (network['network.in.bytes'] + network['network.out.bytes']))*100
    network['out_pct'] = (network['network.out.bytes'] / (network['network.in.bytes'] + network['network.out.bytes']))*100
    network['timestamp'] = network.index
    network = network.dropna()


option = st.selectbox(
    'Chose a server',
    network.node.unique())

st.line_chart(network[network['node'] == option]['in_pct'])

st.dataframe(network)
st.subheader('Exploratory Analysis')

st.markdown('To understand the data we first wrote a function called `plot_network_info`, which shows us the data of each of the nodes in a time series. This was usefull for we were')
st.markdown('Able to see periodicity and see some clear examples of outliers. After that function came a lineplot comparison where we utilize seabron to visualize the change in time')
st.markdown('of all the nodes against each other and can start to get a sense of wheter it\'s related or not.' )

c = alt.Chart(network).mark_line().encode(
    x='timestamp',
    y='in_pct',
    color='node:N')

st.altair_chart(c, use_container_width=True)

st.subheader('Anomaly detection')

st.markdown('For anomaly detection we decided to apply a concept that microsoft utilizes with their own servers for anomaly detection. First we imported a libreary since the functions')
st.markdown('Where already written for us to use. The code base for the function isn\'t to long, but it\'s still better to import than to copy.')
st.markdown('The function we utilize is based on spectral anomaly detection via use of CNNs. The main role is to change our data utilizing a fourier transform, allowing the CNN to understand')
st.markdown('what is happening to our 1 dimensional data. Afterwards we get an output for wheter the CNN believes the input is a anomaly based on the probability that it is. We chose to')
st.markdown('set that any value above 99 percent certainty is considered an outlier. The paper from which we got the idea is https://arxiv.org/pdf/1906.03821.pdf. This probability is then')
st.markdown('returned to us as a score, where the higher the number the more likely it is an anomaly.')
st.markdown('After seeing that the algorithm worked we then created a function to not only calculate our score for the data point, but also to plot the results and see wheter it looks')
st.markdown('correct.')

st.markdown('The formula for this algorithm is based on the following sequence for **x**')
formula_latex = r'''
$$
\begin{equation*}
    \begin{align*}
        A(f) = \text{Amplitude}((\mathscr{F}(\textrm{x}))) \\
        P(f) = \text{Phrase}((\mathscr{F}(\textrm{x}))) \\
        L(f) = log(A(f)) \\
        AL(f) = h_q(f) \cdot L(f)
        R(f) = L(f) - AL(f)
        S(\textrm{x}) = || \mathscr{F}^{-1} (exp(R(f)+iP(f)))||
    \end{align*}
\end{equation*}
$$
'''

st.markdown(formula_latex)

with st.spinner("Ajustando modelo, espera..."):
    # less than period
    amp_window_size=24
    # (maybe) as same as period
    series_window_size=24
    # a number enough larger than period
    score_window_size=100
    spec = anom.Silency(amp_window_size, series_window_size, score_window_size)

    #! Code below can be implemented once button functonality is implemented
    # if node:
    #     test_signal = df[df['node'] == node][network]
    # else:
    #     test_signal = df[network]

    # if timerange:
    #     test_signal = test_signal[(test_signal.index > timerange[0]) & (test_signal.index < timerange[1])]
    

    score_in = spec.generate_anomaly_score(network['in_pct'])
    score_out = spec.generate_anomaly_score(network['out_pct'])

    index_changes_in = np.where(score_in > np.percentile(score_in, 99))[0]
    index_changes_out = np.where(score_out > np.percentile(score_out, 99))[0]



d = alt.Chart(network[network['node'] == option], title="Percentage of incoming traffic").mark_line().encode(
    x='timestamp',
    y='in_pct'
)

e = alt.Chart(network[network['node'] == option], title="Percentage of outgoing traffic").mark_line().encode(
    x='timestamp',
    y='out_pct'
)    


f = alt.Chart(network.iloc[index_changes_in][network['node'] == option]).mark_circle(size=60).encode(
    x = 'timestamp',
    y = 'in_pct',
    color=alt.value('red')
)

st.altair_chart(d+f, use_container_width=True)

g = alt.Chart(network.iloc[index_changes_out][network['node'] == option]).mark_circle(size=60).encode(
    x = 'timestamp',
    y = 'out_pct',
    color=alt.value('red')
)

st.altair_chart(e+g, use_container_width=True)