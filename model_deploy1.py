import streamlit as st
import pandas as pd
import joblib

model= joblib.load(r"L:\Softwares\VS Code\Files\logistic_regression.pkl")
scaler= joblib.load(r"L:\Softwares\VS Code\Files\scaler.pkl")
expected_columns= joblib.load(r"L:\Softwares\VS Code\Files\columns.pkl")

# The exact column order the scaler was fit on (from my notebook, cell 52)
scaler_cols= ["num_units", "injuries_total", "injuries_fatal", "injuries_incapacitating", "injuries_non_incapacitating", "injuries_reported_not_evident", "injuries_no_indication", "crash_hour", "crash_day_of_week", "crash_month", "crash_year", "crash_day"]

st.title("Crash Severity Prediction by Prajjwal Paudel")
st.markdown("Provide the details below")

traffic_control_device= st.selectbox("Traffic Control Device",["TRAFFIC SIGNAL", "STOP SIGN/FLASHER", "NO CONTROLS", "UNKNOWN", "OTHER", "YIELD", "PEDESTRIAN", "CROSSING SIGN", "OTHER REG. SIGN", "LANE USE MARKING", "FLASHING CONTROL SIGNAL", "POLICE/FLAGMAN", "OTHER WARNING SIGN", "RAILROAD CROSSING GATE", "SCHOOL ZONE", "OTHER RAILROAD CROSSING", "RR CROSSING SIGN", "DELINEATORS", "NO PASSING", "BICYCLE CROSSING SIGN"])
damage= st.selectbox("Damage in Property", ["OVER $1,500", "$501 - $1,500", "$500 OR LESS"])
weather_condition= st.selectbox("Weather Condition", ["CLEAR", "RAIN", "CLOUDY/OVERCAST", "SNOW", "UNKNOWN", "OTHER", "FREEZING RAIN/DRIZZLE", "FOG/SMOKE/HAZE", "SLEET/HAIL", "BLOWING SNOW", "SEVERE CROSS WIND GATE", "BLOWING SAND, SOIL, DIRT"])
lighting_condition= st.selectbox("Lighting Condition", ["DAYLIGHT", "DARKNESS, LIGHTED ROAD", "DARKNESS", "DUSK", "UNKNOWN", "DAWN"])
intersection_related_i= st.selectbox("Intersection Related Crash?", ["Y", "N"])
alignment= st.selectbox("Alignment", ["STRAIGHT AND LEVEL", "STRAIGHT ON GRADE", "CURVE, LEVEL", "STRAIGHT ON HILLCREST", "CURVE ON GRADE", "CURVE ON HILLCREST"])
roadway_surface_cond= st.selectbox("Road Surface Condition", ["DRY", "WET", "UNKNOWN", "SNOW OR SLUSH", "ICE", "OTHER", "SAND, MUD, DIRT"])
trafficway_type= st.selectbox("Trafficway Type", ["NOT DIVIDED", "FOUR WAY", "DIVIDED - W/MEDIAN (NOT RAISED)", "ONE-WAY", "DIVIDED - W/MEDIAN BARRIER", "T-INTERSECTION", "OTHER", "CENTER TURN LANE", "UNKNOWN INTERSECTION TYPE", "FIVE POINT, OR MORE", "UNKNOWN", "Y-INTERSECTION", "TRAFFIC ROUTE", "ALLEY", "NOT REPORTED", "PARKING LOT", "RAMP", "ROUNDABOUT", "DRIVEWAY", "L-INTERSECTION"])
first_crash_type= st.selectbox("Collision Type", ["TURNING", "ANGLE", "REAR END", "SIDESWIPE SAME DIRECTION", "PEDESTRIAN", "PEDALCYCLIST", "PARKED MOTOR VEHICLE", "FIXED OBJECT", "SIDESWIPE OPPOSITE DIRECTION", "HEAD ON", "REAR TO FRONT", "REAR TO SIDE", "OTHER OBJECT", "OTHER NONCOLLISION", "OVERTURNED", "ANIMAL", "REAR TO REAR", "TRAIN"])
num_units= st.slider("Number of Vehicles Involved", 1,11,1)
road_defect= st.selectbox("Road Defect at the Crash Location", ["NO DEFECTS", "UNKNOWN", "WORN SURFACE", "OTHER", "RUT, HOLES", "SHOULDER DEFECT", "DEBRIS ON ROADWAY"])
crash_month= st.slider("Crash Occuring Month in Number: For Example (Jaunuary=1)", 1,12,1)
crash_hour = st.slider("Crash Occurring Hour (0-23)", 0, 23, 12)
crash_day_of_week = st.slider("Crash Day of Week (1=Sunday ... 7=Saturday)", 1, 7, 1)
crash_year = st.number_input("Crash Year", min_value=2000, max_value=2100, value=2026, step=1)
crash_day = st.slider("Crash Day of Month", 1, 31, 1)

if st.button("Predict"):

    # ---- Scale only the numeric columns that the scaler actually knows about ----
    # (the scaler was fit on 12 specific columns in the notebook - injuries_* columns
    # aren't collected here since they were dropped from the final feature set, so we
    # only need to manually scale the numeric inputs that remain in X)
    
    def scale_value(col_name, raw_value):
        idx = scaler_cols.index(col_name)
        return (raw_value - scaler.mean_[idx]) / scaler.scale_[idx]
 
    num_units_scaled = scale_value("num_units", num_units)
    crash_month_scaled = scale_value("crash_month", crash_month)
    crash_hour_scaled = scale_value("crash_hour", crash_hour)
    crash_day_of_week_scaled = scale_value("crash_day_of_week", crash_day_of_week)
    crash_year_scaled = scale_value("crash_year", crash_year)
    crash_day_scaled = scale_value("crash_day", crash_day)

    raw_input= {
        # numeric (scaled) features
        "num_units": num_units_scaled,
        "crash_month": crash_month_scaled,
        "crash_hour": crash_hour_scaled,
        "crash_day_of_week": crash_day_of_week_scaled,
        "crash_year": crash_year_scaled,
        "crash_day": crash_day_scaled,
 
        # binary (mapped, NOT one-hot) features
        "is_intersection_crash": 1 if intersection_related_i == "Y" else 0,
 
        # one-hot encoded features - keys must match "original_column_value" exactly
        "traffic_control_device_" + traffic_control_device: 1,
        "damage_" + damage: 1,
        "weather_condition_" + weather_condition: 1,
        "lighting_condition_" + lighting_condition: 1,
        "first_crash_type_" + first_crash_type: 1,
        "alignment_" + alignment: 1,
        "roadway_surface_cond_" + roadway_surface_cond: 1,
        "trafficway_type_" + trafficway_type: 1,
        "road_defect_" + road_defect: 1,
    }

    input_df= pd.DataFrame([raw_input])

    # Fill in every column the model expects but that isn't in this row
    # (this naturally handles: (a) the "drop_first" baseline category for each
    # categorical field, which has no dummy column at all, and (b) any dummy
    # category not selected by the user)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

            # Keep only the columns the model was trained on, in the right order

    input_df= input_df[expected_columns]

    # NOTE: no scaler.transform(input_df) call here - the scaler was fit on a
    # different, smaller set of columns than X. Numeric values were already
    # scaled by hand above using the scaler's stored mean_/scale_.
    prediction= model.predict(input_df) [0]

    if prediction==4:
        st.error("FATAL")
    elif prediction==3:
        st.error("INCAPACITATING INJURY")
    elif prediction==2:
        st.error("REPORTED, NOT EVIDENT")
    elif prediction==1:
        st.error("NON-INCAPACITATING INJURY")
    else:
        st.success("NO INDICATION OF INJURY")        