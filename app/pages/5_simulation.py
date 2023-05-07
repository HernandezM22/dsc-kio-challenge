import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt
import time
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go


st.title("Server behavior simulation and real time anomaly detection")

st.subheader("CPU")

with st.spinner("Generando datos, espera..."):
    cpu = pd.read_pickle("app/data/cpu.pkl")
    cpu["timestamp"] = cpu["timestamp"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
    cpu = cpu.resample('1min', on="timestamp").agg("first").ffill()
    cpu["idle_st"] = (cpu["cpu.idle.pct"]/cpu["cpu.cores"])*100
    cpu["user_st"] = (cpu["cpu.user.pct"]/cpu["cpu.cores"])*100
    cpu["system_st"] = (cpu["cpu.system.pct"]/cpu["cpu.cores"])*100
    cpu["steal_st"] = (cpu["cpu.steal.pct"]/cpu["cpu.cores"])*100
    cpu["user_delta"] = cpu["user_st"] - cpu["user_st"].shift()
    cpu["idle_delta"] = cpu["idle_st"] - cpu["idle_st"].shift()
    cpu["system_delta"] = cpu["system_st"] - cpu["system_st"].shift()
    cpu2 = cpu[cpu["user_delta"].notna()]
    X = cpu2[['user_delta']].values
    model = IsolationForest(n_estimators=1000, contamination=0.003)
    model.fit(X)
    anomaly_scores = model.decision_function(X)
    cpu2['anomaly_score'] = anomaly_scores
    
    pct_kde = KernelDensity(kernel="gaussian")
    pct_kde.fit(cpu["user_st"].values.reshape(-1,1))
    cpu_sample = list(pct_kde.sample(1000))
    cpu_sim = pd.DataFrame(cpu_sample, columns=["samples"])

placeholder = st.empty()
# near real-time / live feed simulation
for seconds in range(1000):
    with placeholder.container():
        df = pd.DataFrame(list(cpu_sim["samples"][0:seconds]), columns=["sim"])
        if len(df) < 2:
            fig = px.line(data_frame=df, y="sim")
            st.plotly_chart(fig, use_container_width=True)
            continue
        else:
            df["anomaly_scores"] = model.decision_function(df[["sim"]].values)
            df["is_anomaly"] = model.predict(df[["sim"]].values)
            st.write(df)
            fig = px.line(data_frame=df, y="sim")
            fig2 = px.scatter(data_frame=df[df["is_anomaly"]== -1], y="sim",  color_discrete_sequence=['red'])
            A = go.Figure(data= fig.data + fig2.data)
            st.plotly_chart(A, use_container_width=True)
    
    time.sleep(0.1)




