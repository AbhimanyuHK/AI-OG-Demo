# Import Module
import pandas as pd
import pydeck as pdk
import streamlit as st

df_store = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Store")
df_terminal = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Terminal")

route_df = pd.read_json("data/optimal_routes.json")
route_df["path"] = route_df["path"].apply(lambda x: [a[::-1] for a in x[0]])

route_df["color"] = [
    tuple(int(x.strip("#")[i: i + 2], 16) for i in (0, 2, 4)) for x in
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
]
print(route_df)

st.set_page_config(
    page_title="Route Optimization",
    page_icon="👋",
)

st.title("Supply Chain: Route Optimization")

st.write("* Transportation Route Optimization from Store to Terminal")

initial_view_state = pdk.ViewState(
    latitude=12.920324579428431,
    longitude=77.59964781527509,
    # longitude=12.920324579428431,
    # latitude=77.59964781527509,
    zoom=12,
    # pitch=50,
)
layers = [
    pdk.Layer(
        'HexagonLayer',
        data=df_store,
        get_position='[lon, lat]',
        radius=200,
        # elevation_scale=4,
        # elevation_range=[0, 1000],
        # pickable=True,
        # extruded=True,
    ),
    pdk.Layer(
        'ScatterplotLayer',
        data=df_terminal,
        get_position='[lon, lat]',
        get_color='[200, 30, 0, 160]',
        get_radius=200,
    ),
    pdk.Layer(
        type="PathLayer",
        data=route_df,
        pickable=True,
        get_color="color",
        width_scale=20,
        width_min_pixels=2,
        get_path="path",
        get_width=2,
    )
]
map_box = pdk.Deck(
    map_style=None,
    layers=layers,
    initial_view_state=initial_view_state,
    tooltip={"text": "{name}"}
)
st.pydeck_chart(map_box)
# map_box.to_html("map_box_route.html")
