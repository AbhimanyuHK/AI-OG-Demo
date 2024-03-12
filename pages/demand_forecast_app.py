from datetime import datetime

import streamlit as st

from forecasting_with_tensorflow import DemandForecastingModel, ModelData

df_model = DemandForecastingModel()
df_model.main()

st.set_page_config(
    page_title="Demand Forecasting Model",
    page_icon="👋",
)

st.title("Demand Forecasting Model")

st.subheader('', divider='rainbow')
col1, col2 = st.columns(2)
col1.metric("Train RMSE", f"{ModelData.TRAIN_RMSE:10.4f}")
col2.metric("Test RMSE", f"{ModelData.TEST_RMSE:10.4f}")

chart_data = ModelData.to_df()

store_option = st.selectbox('Select the Store', set(chart_data["Store"]))

chart_data = chart_data[chart_data["Store"] == store_option]

date_option = st.date_input("When's your birthday", datetime.now().date())

st.line_chart(
    chart_data, x="Date", y=["Actual", "Train", "Test"], color=["#FF0000", "#0000FF", "#00FF00"]  # Optional
)

if date_option:
    print(store_option, date_option.strftime('%Y%m%d'))
    result = df_model.predict(store_option, date_option.strftime('%Y%m%d'))
    # print(result)
    result = result.astype(int)
    result = result.astype(str)
    st.subheader(f'Forecasted Demand for {str(date_option)}', divider='rainbow')
    st.dataframe(result)
