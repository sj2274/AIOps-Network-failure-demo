
import streamlit as st
import pandas as pd, numpy as np, time
from datetime import datetime

st.set_page_config(page_title='AIOps Mini Demo — ATM Network', layout='wide')

st.markdown("# AIOps Mini Demo — ATM Network Failure Risk")
st.markdown("This lightweight demo simulates ATM / Payment Gateway logs and shows **real-time** risk scores using a simple heuristic model. "
            "It's designed for demos and presentations (no external services required).")

# Sidebar controls
st.sidebar.header("Simulation Controls")
msg_rate = st.sidebar.slider("Messages per update", 1, 200, 25)
update_interval = st.sidebar.slider("Update interval (seconds)", 1, 10, 2)
show_stream = st.sidebar.checkbox("Show raw stream", True)
simulate_outage = st.sidebar.checkbox("Inject occasional outage spikes", True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=['time','atm','region','response_time_ms','success','risk'])

col1, col2 = st.columns([3,1])
with col2:
    if st.button("Start Simulation"):
        st.session_state['running'] = True
    if st.button("Stop Simulation"):
        st.session_state['running'] = False
    st.write("Running:" , st.session_state.get('running', False))

def gen_record():
    atm = f"ATM-{np.random.randint(1,150)}"
    region = np.random.choice(["North","South","East","West"])
    base_rt = np.random.normal(200, 35)
    # consider occasional high latency
    if simulate_outage and np.random.random() < 0.02:
        rt = base_rt * np.random.uniform(3,6)
        success = 0
    else:
        rt = base_rt
        success = 1 if np.random.random() > 0.02 else 0
    # simple heuristic risk score
    risk = (min(1.0, (rt/800.0))) + (0 if success==1 else 0.4)
    risk = round(min(1.0, risk), 3)
    return {'time': datetime.now(), 'atm': atm, 'region': region, 'response_time_ms': round(rt,2), 'success': success, 'risk': risk}

# Run simulation loop (non-blocking via session_state)
if 'running' not in st.session_state:
    st.session_state['running'] = False

if st.session_state['running']:
    # generate multiple messages per update
    for _ in range(msg_rate):
        rec = gen_record()
        st.session_state['data'] = pd.concat([st.session_state['data'], pd.DataFrame([rec])], ignore_index=True)
    # limit size
    if len(st.session_state['data']) > 5000:
        st.session_state['data'] = st.session_state['data'].iloc[-3000:].reset_index(drop=True)
    time.sleep(update_interval)

# Layout: Recent stream, Risk leaderboard, Charts
left, right = st.columns([2,1])

with left:
    st.subheader("Recent Events (live)")
    if show_stream:
        st.dataframe(st.session_state['data'].sort_values('time', ascending=False).head(200))
    st.subheader("Response Time (last 200)")
    if not st.session_state['data'].empty:
        st.line_chart(st.session_state['data'].set_index('time')['response_time_ms'].tail(200))
    else:
        st.write("No data yet. Start the simulation.")

with right:
    st.subheader("Top ATMs by Risk (latest)")
    if not st.session_state['data'].empty:
        latest = st.session_state['data'].groupby('atm').last().reset_index()
        top = latest.sort_values('risk', ascending=False).head(10)[['atm','region','response_time_ms','success','risk']]
        st.table(top)
    else:
        st.write("No data yet. Start the simulation.")

st.subheader("Aggregate Metrics")
if not st.session_state['data'].empty:
    agg = st.session_state['data'].resample('1Min', on='time').agg({'response_time_ms':'mean','success':'mean','risk':'mean'}).dropna()
    st.line_chart(agg[['response_time_ms','risk']])
    st.metric("Current Average Uptime (%)", f"{round(100*st.session_state['data']['success'].mean(),2)}%")
    st.metric("Average Risk (last)", round(st.session_state['data']['risk'].tail(50).mean(),3))
else:
    st.write("No aggregate metrics yet.")

st.markdown("---")
st.markdown("**Notes:** This mini-demo uses a simple heuristic risk formula for demonstration. For a real project, replace the heuristic with trained models (LSTM, Isolation Forest) and integrate with Kafka / ELK / Prometheus as described in the full report.")
st.markdown("**To run locally:** `pip install streamlit pandas numpy` → `streamlit run app.py`")
