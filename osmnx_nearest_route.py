import osmnx as ox
import pandas as pd

ox.settings.use_cache = True
ox.settings.log_console = True

# Import Module


df_store = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Store")
df_terminal = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Terminal")

df_plan = pd.read_csv("data/demand_and_supply_plan.csv")
df_plan = df_plan[df_plan["supply plan"] != 0]

df = pd.merge(df_plan, df_store, on="store", how="inner")
df = pd.merge(df, df_terminal, on="terminal", how="inner", suffixes=('_s', '_t'))

df["name"] = df["store"] + " - " + df["terminal"]
df["path"] = df.apply(lambda x: [[x["lat_s"], x["lon_s"]], [x["lat_t"], x["lon_t"]]], axis=1)
# print(df)

route_df = df[["name", "path"]]
route_df["color"] = route_df.apply(lambda x: tuple(int("FFFFFF"[i: i + 2], 16) for i in (0, 2, 4)), axis=1)

print(route_df)

# # get graph and add free-flow travel times to its edges
# G = ox.graph_from_place("Bengaluru, Karnataka, India", network_type="drive")
# G = ox.add_edge_travel_times(ox.add_edge_speeds(G))
#
#
# def find_shortest_route(_path):
#     station_org_node_id = ox.distance.nearest_nodes(G, [_path[0][1]], [_path[0][1]])
#     station_dst_node_id = ox.distance.nearest_nodes(G, [_path[1][1]], [_path[1][1]])
#
#     paths = ox.shortest_path(G, station_org_node_id, station_dst_node_id, weight="travel_time", cpus=None)
#
#     paths_latlng = [[(G.nodes[node]["y"], G.nodes[node]["x"]) for node in path] for path in paths]
#     return paths_latlng
#
#
# route_df["path"] = route_df["path"].apply(find_shortest_route)
# route_df.to_json("data/optimal_routes.json")
