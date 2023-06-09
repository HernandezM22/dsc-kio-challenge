import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.ensemble import IsolationForest
import altair as alt
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Memory Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Memory Monitoring")

st.write("After receiving the original .csv file, we decided to separate it by event type")
st.write("Focusing on the events flagged as 'memory', first we divided this dataset by nodes, because not every node has the same behaviour.")
st.write("Having this subsets, we began performing an EDA.")
st.write("We had entries every 10 seconds approximately, we resampled this into the first entry of every minute, now having time frequencies every minute.")


st.write("We focused on analizing the percentage of memory used, specifically the change of this VARIABLE every minute. to detect anomalies. Meaning greater changes in this, would determine the possibility of an existing anomaly.")

st.write("Using unsupervised outlier detection algorithm, we used Isolation Forest. Which focuses on how far or isolated the data is from the rest.")

st.write("For the forecast, we use montecarlo simulation, were the resulting values are the mean of the several generated paths.")

with st.spinner("Cargando, espera..."):
    memory = pd.read_pickle("app/data/memory.pkl")
    memory["timestamp"] = memory["timestamp"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    qaserver = memory[memory['node'] == 'QASERVER']
    #qaserver = qaserver.resample('1min', on="timestamp").agg("first").ffill()
    ## qaserver used
    qaserver['actual_used'] = qaserver['memory.actual.used.pct']
    qaserver = qaserver.sort_values(by=['timestamp'])
    qaserver = qaserver.resample('1Min', on="timestamp").agg("first").ffill()
    # Calculate deltas
    qaserver['delta_used'] = qaserver['memory.actual.used.pct'] - qaserver['memory.actual.used.pct'].shift()

st.subheader("Raw usage")

st.line_chart(qaserver['memory.actual.used.bytes'])

#st.subheader("Standardized usage")

#st.line_chart(cpu["user_st"])

st.subheader("First order differences over time")

st.line_chart(qaserver['delta_used'])

st.subheader("Anomaly detection")

with st.spinner("Ajustando modelo, espera..."):
    qaserver2 = qaserver[qaserver['delta_used'].notna()]
    X = qaserver2[['delta_used']].values
    model = IsolationForest(n_estimators=1000, contamination=0.003)
    model.fit(X)
    anomaly_scores = model.decision_function(X)
    anomalies = model.predict(X)
    qaserver2['anomaly_score'] = anomaly_scores
    qaserver2['is_anomaly'] = anomalies
    qaserver2['timestamp'] = qaserver2.index

c = alt.Chart(qaserver2).mark_line().encode(
    x='timestamp',
    y='delta_used')
d = alt.Chart(qaserver2[qaserver2['is_anomaly'] == -1]).mark_point(color='red', size=50).encode(
    x='timestamp',
    y='delta_used'
)

# fig, x = plt.subplots()

# x.plot(qaserver2['timestamp'], qaserver2['delta_used'])
# x.scatter(qaserver2[qaserver2['is_anomaly'] == -1]['timestamp'], qaserver2[qaserver2['is_anomaly'] == -1]['delta_used'], c='red')
# st.pyplot(fig)
st.altair_chart(c+d, use_container_width=True)

with st.spinner("Realizando predicciones, espera..."):
    
    X = qaserver2['memory.actual.used.pct'].values.reshape(-1, 1)

    # Fit a kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(X)

    # Generate n trajectories from the fitted distribution
    n = 1000
    trajectories_list = []
    for i in range(100):
        trajectory = list(kde.sample(100))
        trajectories_list.append(trajectory)
    
    timestamps = pd.date_range(start=qaserver2['timestamp'].max(), periods=100, freq='1min')

    # Append the generated trajectories and timestamps to the original dataframe
    new_df = pd.DataFrame(data=trajectories_list, columns=[f'traj{i+1}' for i in range(100)])
    new_df['timestamp'] = timestamps

    new_df["mean_traj"] = new_df.mean(axis=1)


f = alt.Chart(qaserver2[-100:]).mark_line().encode(
    x='timestamp',
    y='actual_used')


g = alt.Chart(new_df).mark_line(color="red").encode(
    x='timestamp',
    y='mean_traj')

st.altair_chart(f+g, use_container_width=True)