import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from mpl_interactions import zoom_factory
from datetime import datetime, timedelta

# Load the trained model
def load_model():
    with open('decision_tree_regressor.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Load the dataset directly with 'Days'
def load_data():
    data = pd.read_csv("cleaned_data.csv")
    return data

def convert_days_to_date(days):
    start_date = datetime(2004, 1, 2)
    return start_date + timedelta(days=int(days))

# Visualization function
def visualize_predict(data, start_day, end_day):
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Decision Tree Analysis')
    ax.set_xlabel('Days (Since 2004-01-02)')
    ax.set_ylabel('Closed Price ($USD)')

    ax.plot(data['Days'].iloc[start_day:end_day], data['Close'].iloc[start_day:end_day], label='Actual Close Price')
    if start_day <= len(data) - 21 <= end_day:
        ax.plot(data['Days'].iloc[-21:], data['Predictions'].iloc[-21:], label='Predicted Close Price', color='red')
    ax.legend()

    zoom_factory(ax)

    return fig

# Main app function
def main():
    st.title("NVIDIA Stock Price Prediction App")

    with st.expander("Project Overview"):
        st.write("""
            This project employs a Decision Tree Regressor to forecast the closing prices of NVIDIA stocks for the next 21 days. By analyzing historical stock data, the model identifies trends and patterns, providing valuable insights for informed investment decisions.
        """)

    with st.expander("Methods Used"):
        st.write("""
            Data Cleaning: Leveraging Python libraries like Pandas to clean the dataset for analysis.\n
            Data Visualization: Employing Matplotlib and Seaborn to create visual representations of the data, allowing for the exploration of potential relationships and trends.\n
            Model Training: Utilizing scikit-learn to train the Decision Tree Regressor model.\n
            Model Evaluation: Assessing the model's performance through scikit-learn's mean squared error.\n
            Model Deployment: Saving the trained model with Pickle for future use.
        """)

    with st.expander("For Transparency Purposes"):
        st.write("""In the course of the debugging process, I leveraged Perplexity AI to assist with certain aspects of problem identification and resolution. This tool facilitated a more efficient analysis and helped streamline the troubleshooting efforts.""")

    with st.expander("DISCLAIMER"):
        st.write("""The stock predictions provided here are based on analytical models and historical data. Please remember that the stock market can be unpredictable, influenced by various factors and unforeseen events. As such, these predictions should be taken with caution and are only intended for entertainment purposes only. Always do your own research and consider talking to a financial advisor before making any investment decisions.""")

    # Load the data
    data = load_data()

    # Display the dataset
    st.subheader("NVIDIA Stock Data")
    st.dataframe(data)

    # Display the sidebar with various visualization options
    st.sidebar.header("Visualizations")
    plot_options = ["Predicted Plot"]
    selected_plot = st.sidebar.selectbox("Select a plot type", plot_options)

    # Create a slider for number of trading days
    start_day = st.sidebar.slider("Select starting trading day", 0, len(data) - 500, 0)
    end_day = st.sidebar.slider("Select ending trading day", start_day + 1, len(data) - 1, start_day + 500)

    if selected_plot == "Predicted Plot":
        # Load the model and make predictions for the last 21 days
        model = load_model()

        # Create future predictions
        future_pred = data[['Days', 'Open', 'High', 'Low', 'Close', 'Volume']].iloc[-2 * 21:-21]
        predictions = model.predict(future_pred)

        # Initialize the Predictions column with NaNs and insert predictions at the end
        data['Predictions'] = [None] * len(data)
        data.loc[len(data) - 21:, 'Predictions'] = predictions  # Set predictions in the last 21 rows

        # Visualize the predictions
        st.subheader("Decision Tree Analysis Visualization")
        fig = visualize_predict(data, start_day, end_day)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # Date Converter Section
    st.subheader("Date Converter")
    st.write("Convert days to dates")

    col1, col2 = st.columns(2)

    with col1:
        start_day_input = st.number_input("Start Day", min_value=int(data['Days'].min()), max_value=int(data['Days'].max()), value=int(data['Days'].min()))
        start_date_converted = convert_days_to_date(start_day_input)
        st.write(f"Start Date: {start_date_converted.strftime('%Y-%m-%d')}")

    with col2:
        end_day_input = st.number_input("End Day", min_value=int(data['Days'].min()), max_value=int(data['Days'].max()), value=int(data['Days'].max()))
        end_date_converted = convert_days_to_date(end_day_input)
        st.write(f"End Date: {end_date_converted.strftime('%Y-%m-%d')}")

    if end_day_input < start_day_input:
        st.warning("End Day cannot be before Start Day. Please adjust your selection.")

    st.write(f"Selected range: {end_day_input - start_day_input + 1} days")

if __name__ == "__main__":
    main()


