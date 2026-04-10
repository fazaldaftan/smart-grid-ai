import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import pickle
import requests
from datetime import datetime
import openmeteo_requests
from retry_requests import retry
import requests_cache
import matplotlib.pyplot as plt
import shap
from streamlit_geolocation import streamlit_geolocation
import streamlit.components.v1 as components

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="SmartGrid Enterprise", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0px 0px; padding-top: 10px; padding-bottom: 10px; font-weight: 600; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# 2. HELPER FUNCTIONS 
@st.cache_data(ttl=3600)
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    try:
        response = requests.get(url).json()
        if "results" in response and len(response["results"]) > 0:
            loc = response["results"][0]
            return loc["latitude"], loc["longitude"], loc["name"], loc.get("country", "")
    except: pass
    return None, None, None, None

@st.cache_resource
def load_models():
    try: 
        xgb = joblib.load('xgb_model.pkl')
        explainer = joblib.load('shap_explainer_live.pkl')
        with open('feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return xgb, explainer, feature_cols
    except Exception as e:
        st.error(f"⚠️ CRITICAL MODEL ERROR: {e}")
        st.info("Check if all .pkl files are in the same folder as app.py")
        return None, None, None

def engineer_features(df, feature_cols):
    df = df.copy()
    df['hour'], df['dayofweek'], df['month'] = df.index.hour, df.index.dayofweek, df.index.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    for col, period in [('hour', 24), ('dayofweek', 7), ('month', 12)]:
        df[f'{col}_sin'], df[f'{col}_cos'] = np.sin(2 * np.pi * df[col] / period), np.cos(2 * np.pi * df[col] / period)
    
    df['temp_lag1'] = df['temperature_2m'].shift(1).bfill()
    df['hdd'] = np.maximum(18.0 - df['temperature_2m'], 0)
    df['cdd'] = np.maximum(df['temperature_2m'] - 18.0, 0)
    df['cloud_impact'] = df['cloud_cover'] / 100.0
    df['effective_radiation'] = df['shortwave_radiation'] * (1 - df['cloud_impact'])
        
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    return df[feature_cols].astype(np.float32)

def generate_smart_schedule(forecast, weather_df, current_solar_kw, current_batt_cap):
    hours = [datetime.now().replace(minute=0, second=0) + pd.Timedelta(hours=i) for i in range(24)]
    data = []
    current_battery = 0.0 
    
    for i, (dt, demand) in enumerate(zip(hours, forecast)):
        h = dt.hour
        price = 7.47 if h < 6 else 12.45 if h < 9 else 9.96 if h < 16 else 23.24 if h < 21 else 9.13
        
        solar = 0
        if 'shortwave_radiation' in weather_df.columns:
            rad = weather_df['shortwave_radiation'].iloc[i] if i < len(weather_df) else 0
            solar = min(current_solar_kw, rad / 1000 * current_solar_kw * 0.75) if current_solar_kw > 0 else 0
            
        net_demand_before_batt = demand - solar
        batt_charge, batt_discharge = 0, 0
        
        if net_demand_before_batt < 0: 
            batt_charge = min(abs(net_demand_before_batt), current_batt_cap - current_battery)
            current_battery += batt_charge
            net_grid = 0
        else: 
            if price >= 15: 
                batt_discharge = min(net_demand_before_batt, current_battery)
                current_battery -= batt_discharge
            net_grid = net_demand_before_batt - batt_discharge
            
        rec = "🛑 Peak Avoidance" if price == 23.24 else "☀️ Charging Battery" if batt_charge > 0 else "🔋 Discharging" if batt_discharge > 0 else "✅ Normal"
        data.append({'Hour': dt.strftime('%H:%M'), 'Demand_kWh': round(demand, 2), 'Solar_kWh': round(solar, 2), 'Battery_Level': round(current_battery, 2), 'Grid_Draw_kWh': round(net_grid, 2), 'Price_Unit': price, 'Action': rec})
    return pd.DataFrame(data)

# 3. SIDEBAR: PROFESSIONAL INPUTS & AGENCY SETTINGS
with st.sidebar:
    with st.expander("⚙️ B2B Agency Settings (White-Label)"):
        agency_name = st.text_input("Platform Name", value="SmartGrid AI")
        currency = st.selectbox("Display Currency", ["₹", "$", "€", "£"])
        solar_cost_per_kw = st.number_input(f"Cost per kW Solar ({currency})", value=5000)
        batt_cost_per_kwh = st.number_input(f"Cost per kWh Battery ({currency})", value=4000)

    st.header("🌍 Location Settings")
    search_mode = st.radio("Choose Location Method:", ["Search by City", "Use Live GPS"])
    
    lat, lon, resolved_city, country = None, None, "", ""
    
    if search_mode == "Use Live GPS":
        loc = streamlit_geolocation()
        if loc and loc.get('latitude') is not None and loc.get('longitude') is not None:
            lat, lon = loc['latitude'], loc['longitude']
            resolved_city = "Current GPS Location"
            st.success("✅ GPS Locked")
    else:
        city_input = st.text_input("Enter City Name", value="Pune")
        lat, lon, resolved_city, country = get_coordinates(city_input)
        if lat: st.success(f"✅ Found: {resolved_city}, {country}")

    if lat is not None and lon is not None:
        fig_globe = go.Figure(go.Scattergeo(
            lon=[lon], lat=[lat], mode='markers+text', text=["📍 Target Area"], textposition="bottom center",
            marker=dict(size=14, color='#E53935', symbol='circle', line=dict(width=2, color='white'))
        ))
        fig_globe.update_geos(projection_type="orthographic", showcoastlines=True, showland=True, landcolor="#121212", showocean=True, oceancolor="#0A192F", bgcolor="rgba(0,0,0,0)")
        fig_globe.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_globe, use_container_width=True, config={'displayModeBar': False})
    
    st.divider()
    st.header("🏠 Live Asset Profile")
    num_people = st.slider("Number of residents", 1, 10, 4)
    house_size = st.number_input("House size (m²)", 50, 500, 120, step=10)
    has_ac = st.toggle("❄️ Air Conditioning", value=True)
    has_ev = st.toggle("🚗 Electric Vehicle (EV)", value=False)
    
    st.subheader("☀️ Energy Independence")
    has_solar = st.toggle("Rooftop Solar", value=True)
    solar_kw = st.slider("Solar Array (kW)", 0.0, 20.0, 5.0) if has_solar else 0.0
    
    has_battery = st.toggle("🔋 Home Battery System", value=True)
    battery_capacity = st.slider("Battery Size (kWh)", 0.0, 30.0, 10.0) if has_battery else 0.0

    st.divider()
    if st.button("🔮 Run Optimization", use_container_width=True, type="primary"):
        if lat is not None:
            st.session_state.fetch_data = True
        else:
            st.error("Please resolve the location first.")

# 4. HEADER
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.title(f"⚡ {agency_name} Optimizer")
    st.caption("**Commercial Grade SaaS** | V3 Enterprise AI Engine • Live Telemetry")

# 5. FETCH WEATHER & TELEMETRY
xgb_model, explainer, feature_cols = load_models()

if st.session_state.get('fetch_data', False):
    if feature_cols is None:
        st.error("❌ Cannot run AI: Models missing.")
        st.session_state.fetch_data = False
    else:
        st.session_state['location_title'] = f"{resolved_city}, {country}" if country else resolved_city
        st.session_state['lat'], st.session_state['lon'] = lat, lon
        
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        openmeteo = openmeteo_requests.Client(session=retry(cache_session, retries=5))
        params = {"latitude": lat, "longitude": lon, "hourly": ["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m", "relative_humidity_2m"], "forecast_days": 2, "timezone": "auto"}
        response = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        hourly = response.Hourly()
        
        time_index = pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True), end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()), inclusive='left')
        weather_data = {"datetime": time_index}
        for i, var in enumerate(["temperature_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m", "relative_humidity_2m"]): 
            weather_data[var] = hourly.Variables(i).ValuesAsNumpy()[:len(time_index)]
        weather_df = pd.DataFrame(weather_data).set_index('datetime').head(24)
        
        st.session_state['weather_df'] = weather_df
        st.session_state['dummy_base'] = pd.DataFrame(index=weather_df.index).join(weather_df)
        st.session_state['app_ready'] = True
        st.session_state['fetch_data'] = False

# 6. REACTIVE AI EXECUTION (Runs instantly when sliders change)
if st.session_state.get('app_ready', False):
    weather_df = st.session_state['weather_df']
    
    # Pass metadata directly into V3 AI
    dummy = st.session_state['dummy_base'].copy()
    dummy['num_people'] = num_people
    dummy['house_size_sqm'] = house_size
    dummy['has_ac'] = int(has_ac)
    dummy['has_ev'] = int(has_ev)
    
    user_feat = engineer_features(dummy, feature_cols)
    final_pred = np.clip(xgb_model.predict(user_feat), 0, None)
    st.session_state['user_feat'] = user_feat
    
    active_solar = solar_kw if has_solar else 0
    active_battery = battery_capacity if has_battery else 0
    schedule = generate_smart_schedule(final_pred, weather_df, active_solar, active_battery)
    
    st.success(f"✅ V3 AI Grid Optimization Active for **{st.session_state['location_title']}**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Forecast", "💼 20-Year Financial ROI", "📡 Weather Mapping", "🧠 AI Explainability"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gross Demand", f"{final_pred.sum():.1f} kWh")
        c2.metric("Solar Generated", f"{schedule['Solar_kWh'].sum():.1f} kWh")
        c3.metric("Net Grid Draw", f"{schedule['Grid_Draw_kWh'].sum():.1f} kWh", delta=f"-{(final_pred.sum() - schedule['Grid_Draw_kWh'].sum()):.1f} kWh avoided", delta_color="inverse")
        c4.metric("Peak Stress", "🔴 EV Warning" if has_ev else "🟢 Nominal")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=schedule['Hour'], y=final_pred, mode='lines', name='Total Home Demand', line=dict(color='#888', dash='dot')))
        fig.add_trace(go.Scatter(x=schedule['Hour'], y=schedule['Grid_Draw_kWh'], mode='lines+markers', name='Actual Grid Draw', line=dict(color='#E53935', width=4)))
        if has_solar: fig.add_trace(go.Bar(x=schedule['Hour'], y=schedule['Solar_kWh'], name='Solar Yield', marker_color='#FDD835', opacity=0.6))
        if has_battery: fig.add_trace(go.Scatter(x=schedule['Hour'], y=schedule['Battery_Level'], mode='lines', name='Battery SoC', fill='tozeroy', line=dict(color='#4CAF50')))
        fig.update_layout(title="24-Hour AI Energy Profile", xaxis_title="Time", yaxis_title="Energy (kWh)", height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        baseline_cost_daily = (schedule['Demand_kWh'] * schedule['Price_Unit']).sum()
        optimized_cost_daily = (schedule['Grid_Draw_kWh'] * schedule['Price_Unit']).sum()
        yearly_baseline = baseline_cost_daily * 365
        yearly_optimized = optimized_cost_daily * 365
        total_capex = (active_solar * solar_cost_per_kw) + (active_battery * batt_cost_per_kwh)
        
        years, status_quo, smart_sys = np.arange(0, 21), [0], [total_capex]
        curr_base, curr_opt = yearly_baseline, yearly_optimized
        
        for y in range(1, 21):
            curr_base *= 1.03
            curr_opt *= 1.03
            status_quo.append(status_quo[-1] + curr_base)
            smart_sys.append(smart_sys[-1] + curr_opt)
            
        payback_year = next((i for i in range(21) if status_quo[i] > smart_sys[i]), None)
        net_profit_20yr = status_quo[-1] - smart_sys[-1]
        
        st.markdown(f"### 💼 {agency_name} 20-Year Investment Proposal")
        f1, f2, f3 = st.columns(3)
        f1.metric("Total Upfront CAPEX", f"{currency}{total_capex:,.0f}")
        f2.metric("Break-Even Point", f"Year {payback_year}" if payback_year else "N/A", "ROI Hit" if payback_year else "Need bigger system")
        f3.metric("20-Year Pure Profit", f"{currency}{net_profit_20yr:,.0f}")
        
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(x=years, y=status_quo, name='Status Quo (Grid Only)', line=dict(color='#E53935', dash='dash')))
        fig_roi.add_trace(go.Scatter(x=years, y=smart_sys, name='Smart System', line=dict(color='#4CAF50', width=4)))
        if payback_year: fig_roi.add_vrect(x0=payback_year, x1=20, fillcolor="rgba(76, 175, 80, 0.1)", layer="below", annotation_text="Profit Zone")
        fig_roi.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_roi, use_container_width=True)

    with tab3:
        st.markdown("#### 🌬️ Live Interactive Wind Map")
        windy_html = f"""<iframe width="100%" height="450" src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=°C&metricWind=km/h&zoom=10&overlay=wind&product=ecmwf&level=surface&lat={st.session_state['lat']}&lon={st.session_state['lon']}" frameborder="0" style="border-radius: 10px;"></iframe>"""
        components.html(windy_html, height=450)

    with tab4:
        st.markdown("### 🧠 AI Feature Impact (SHAP)")
        st.caption("How did the V3 Neural Tree make its decision for Hour 0?")
        if st.button("Generate SHAP Waterfall"):
            with st.spinner("Analyzing parameters..."):
                if explainer is not None:
                    shap_values = explainer.shap_values(st.session_state['user_feat'].iloc[[0]])
                    fig, ax = plt.subplots(figsize=(8, 5))
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=st.session_state['user_feat'].iloc[0], feature_names=feature_cols), max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)