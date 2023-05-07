import streamlit as st
import pandas as pd
import datetime as dt
from sklearn.ensemble import IsolationForest
import altair as alt
from sklearn.neighbors import KernelDensity

# Configure the page setting
st.set_page_config(
    page_title="CPU Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CPU Monitoring")


with st.spinner("Cargando, espera..."):
    cpu = pd.read_pickle("app/data/cpu.pkl") # Load the data
    cpu["timestamp"] = cpu["timestamp"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')) # Set out time column into datetime format
    cpu = cpu.resample('1min', on="timestamp").agg("first").ffill() # Resample our data for easier analysis / implementation
    # Standarize the data from the cpu, as the idle + user + system + steal should equal the number of cores
    cpu["idle_st"] = (cpu["cpu.idle.pct"]/cpu["cpu.cores"])*100
    cpu["user_st"] = (cpu["cpu.user.pct"]/cpu["cpu.cores"])*100
    cpu["system_st"] = (cpu["cpu.system.pct"]/cpu["cpu.cores"])*100
    cpu["steal_st"] = (cpu["cpu.steal.pct"]/cpu["cpu.cores"])*100
    # We calculate the derivative of our variables to see the tangent, or rather, where they are heading.
    cpu["user_delta"] = cpu["user_st"] - cpu["user_st"].shift()
    cpu["idle_delta"] = cpu["idle_st"] - cpu["idle_st"].shift()
    cpu["system_delta"] = cpu["system_st"] - cpu["system_st"].shift()


st.subheader("Raw usage")

st.line_chart(cpu["cpu.user.pct"]) # Create the chart for the percentage of cpu usage of user

st.subheader("Standardized usage")

st.line_chart(cpu["user_st"])

st.subheader("First order differences over time")

st.line_chart(cpu["user_delta"])

st.subheader("Anomaly detection")

with st.spinner("Ajustando modelo, espera..."):
    # We first elliminate nas from our data to be able to classify.
    cpu2 = cpu[cpu["user_delta"].notna()]
    X = cpu2[['user_delta']].values 
    model = IsolationForest(n_estimators=1000, contamination=0.003) # We create our model with initialized variables.
    model.fit(X)
    anomaly_scores = model.decision_function(X)
    anomalies = model.predict(X) 
    # We add our outputs into our dataframe so that we can plot them more easily.
    cpu2['anomaly_score'] = anomaly_scores 
    cpu2['is_anomaly'] = anomalies
    cpu2['timestamp'] = cpu2.index
# We plot the derivative
c = alt.Chart(cpu2).mark_line().encode(
    x='timestamp',
    y='user_delta') 
# Plot the anomalies to check where they are in our timeseries.
d = alt.Chart(cpu2[cpu2['is_anomaly'] == -1]).mark_point(color='red', size=50).encode(
    x='timestamp',
    y='user_delta'
)

st.altair_chart(c+d, use_container_width=True)

with st.spinner("Realizando predicciones, espera..."):
    # We reshape our data so that we can utilize the KernelDensity object    
    X = cpu2["user_delta"].values.reshape(-1, 1)

    # Fit a kernel density estimator
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

    # Generate n trajectories from the fitted distribution
    # We utilize Monte Carlo for simulation
    n = 1000
    trajectories_list = []
    for i in range(100):
        trajectory = list(kde.sample(100))
        trajectories_list.append(trajectory) #We calculate the trajectories of our model
    
    timestamps = pd.date_range(start=cpu2['timestamp'].max(), periods=100, freq='1min') # We standarize to 1 minute to have it the same as in our original data resampled

    # Append the generated trajectories and timestamps to the original dataframe
    new_df = pd.DataFrame(data=trajectories_list, columns=[f'traj{i+1}' for i in range(100)])
    new_df['timestamp'] = timestamps

    new_df["mean_traj"] = new_df.mean(axis=1)


f = alt.Chart(cpu2[-100:]).mark_line().encode(
    x='timestamp',
    y='user_delta')


g = alt.Chart(new_df).mark_line(color="red").encode(
    x='timestamp',
    y='mean_traj')

st.altair_chart(f+g, use_container_width=True)

st.subheader("Per-server view")
option = st.selectbox(
    'Chose a server',
    cpu.node.unique())
st.subheader("Standardized usage")

st.line_chart(cpu[cpu["node"]==option]["user_st"])

st.subheader("First order difference")

st.line_chart(cpu[cpu["node"]==option]["user_delta"])

st.subheader("Anomaly detection")

c = alt.Chart(cpu2[cpu2["node"] == option]).mark_line().encode(
    x='timestamp',
    y='user_delta')

d = alt.Chart(cpu2[(cpu2['is_anomaly'] == -1) & (cpu2["node"]==option)]).mark_point(color='red', size=50).encode(
    x='timestamp',
    y='user_delta'
)

st.altair_chart(c+d, use_container_width=True)

