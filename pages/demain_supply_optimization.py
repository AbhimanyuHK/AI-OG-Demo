import pandas as pd
import streamlit as st
# Import Module
from pulp import *

df = pd.read_excel("data/demand_and_supply.xlsx", sheet_name="Data")
# Initialize Model
qcells_model = LpProblem("Supply Distribution Problem", LpMinimize)

region = ["East", "West", "North", "South"]
demand = [1380, 1815, 920, 460]
regional_demand = dict(zip(region, demand))
warehouse = ["GA", "NJ", "AZ"]
supply = [2075, 1500, 1000]
warehouse_inventory = dict(zip(warehouse, supply))
costs = {
    ("GA", "East"): 85, ("GA", "West"): 65, ("GA", "North"): 45, ("GA", "South"): 20,
    ("NJ", "East"): 35, ("NJ", "West"): 120, ("NJ", "North"): 35, ("NJ", "South"): 75,
    ("AZ", "East"): 110, ("AZ", "West"): 55, ("AZ", "North"): 50, ("AZ", "South"): 50
}

# Decision Variable
logistics = LpVariable.dicts('K', [(i, r) for i in warehouse for r in region], lowBound=0)
# print(logistics)

# Define Objective Function
qcells_model += lpSum([logistics[i, r] * costs[(i, r)] for r in region for i in warehouse])

# Constraint 01
for i in warehouse:
    qcells_model += lpSum([logistics[(i, r)] for r in region]) == warehouse_inventory[i]
# Constraint 02
for r in region:
    qcells_model += lpSum([logistics[(i, r)] for i in warehouse]) == regional_demand[r]

qcells_model.solve()
print(LpStatus[qcells_model.status])

s_t_list = []
for v in qcells_model.variables():
    # print(v.name, "=", v.varValue)
    s, t = v.name.strip().replace("('", "").replace("')", "").replace("'", "").replace(",", "").split("_")[1:]
    s_t_list.append({"store": s, "terminal": t, "supply plan": v.varValue})

# print(s_t_list)
obj_value = qcells_model.objective.value()
# print(obj_value)
# 205675.0


st.set_page_config(
    page_title="Supply Chain Opt",
    page_icon="👋",
)

st.title("Supply Chain: Supply Distribution")

st.write("* Transportation Cost Optimization from Store to Terminal")

col1, col2 = st.columns(2)

col1.subheader(f'Demand and Supply Info', divider='rainbow')

col1.dataframe(df)

col2.subheader(f'Objective Value {obj_value}', divider='rainbow')

d_s_object_df = pd.DataFrame(s_t_list)
d_s_object_df.to_csv("data/demand_and_supply_plan.csv", index=False)
col2.dataframe(d_s_object_df)
