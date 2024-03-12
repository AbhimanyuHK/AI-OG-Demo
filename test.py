# Import Module
import pandas as pd
import pydeck as pdk

df_store = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Store")
df_terminal = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Terminal")

route_df = pd.read_json("data/optimal_routes.json")
route_df["path"] = route_df["path"].apply(lambda x: x[0])
agg_route_paths = route_df[["path"]].to_dict(orient="records")

initial_view_state = pdk.ViewState(
    latitude=12.9369075,
    longitude=77.6110193,
    zoom=11,
    # pitch=50,
)
layer = pdk.Layer(
    'PathLayer',
    agg_route_paths,
    width_min_pixels=5,
    get_color='[255, 147, 0, 150]',
)

layers = [
    layer,
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
    )
]
map_box = pdk.Deck(
    # map_style=None,
    layers=layers,
    initial_view_state=initial_view_state,
    # tooltip={"text": "{name}"}
)
map_box.to_html("map_box_route.html")
