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

with st.spinner("Cargando, espera..."):
    memory = pd.read_pickle("app/data/memory.pkl")
    memory["timestamp"] = memory["timestamp"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    qaserver = memory[memory['node'] == 'QASERVER']
    #qaserver = qaserver.resample('1min', on="timestamp").agg("first").ffill()
    ## qaserver used
    qaserver = qaserver.sort_values(by=['timestamp'])
    qaserver = qaserver.resample('1Min', on="timestamp").agg("first").ffill()
    # Calculate deltas
    qaserver['delta actual.used.pct'] = qaserver['memory.actual.used.pct'] - qaserver['memory.actual.used.pct'].shift()

st.subheader("Raw usage")

st.line_chart(qaserver['memory.actual.used.bytes'])

#st.subheader("Standardized usage")

#st.line_chart(cpu["user_st"])

st.subheader("First order differences over time")

st.line_chart(qaserver['delta actual.used.pct'])

st.subheader("Anomaly detection")

with st.spinner("Ajustando modelo, espera..."):
    qaserver2 = qaserver[qaserver['delta actual.used.pct'].notna()]
    X = qaserver2[['delta actual.used.pct']].values
    model = IsolationForest(n_estimators=1000, contamination=0.003)
    model.fit(X)
    anomaly_scores = model.decision_function(X)
    anomalies = model.predict(X)
    qaserver2['anomaly_score'] = anomaly_scores
    qaserver2['is_anomaly'] = anomalies
    qaserver2['timestamp'] = qaserver2.index

# c = alt.Chart(qaserver2).mark_line().encode(
#     x='timestamp',
#     y='delta actual.used.pct')

# d = alt.Chart(qaserver2[qaserver2['is_anomaly'] == -1]).mark_point(color='red', size=50).encode(
#     x='timestamp',
#     y='delta actual.used.pct'
# )

fig, x = plt.subplots()

x.plot(qaserver2['timestamp'], qaserver2['delta actual.used.pct'])
x.scatter(qaserver2[qaserver2['is_anomaly'] == -1]['timestamp'], qaserver2[qaserver2['is_anomaly'] == -1]['delta actual.used.pct'], c='red')
st.pyplot(fig)
#st.altair_chart(c+d, use_container_width=True)

with st.spinner("Realizando predicciones, espera..."):
    
    X = qaserver2['delta actual.used.pct'].values.reshape(-1, 1)

    # Fit a kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

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

st.dataframe(new_df)


f = alt.Chart(qaserver2[-100:]).mark_line().encode(
    x='timestamp',
    y='delta actual.used.pct')


g = alt.Chart(new_df).mark_line(color="red").encode(
    x='timestamp',
    y='mean_traj')

st.altair_chart(f+g, use_container_width=True)