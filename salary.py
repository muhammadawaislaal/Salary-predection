import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt

# Set up the page
st.set_page_config(page_title="Advanced Salary Predictor", layout="wide")
st.title("💰Salary Prediction Dashboard")
st.markdown("Predict salaries based on multiple professional attributes")

# Custom function to safely load model
def safe_load_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError()])
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Sidebar for user controls
with st.sidebar:
    st.header("⚙️ Input Parameters")
    years_exp = st.slider("Years of Experience", 
                         min_value=0.0, 
                         max_value=30.0, 
                         value=5.0, 
                         step=0.5)
    
    age = st.slider("Age", 
                   min_value=18, 
                   max_value=70, 
                   value=30)
    
    position = st.selectbox("Position Level",
                          ["lead generator", "graphic designer", "webdeveloper", "Ai Engineer", "Executive"])
    
    qualification = st.selectbox("Highest Qualification",
                               ["High School", "Bachelor's", "Master's", "PhD"])
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.write("This app uses a trained Artificial Neural Network to predict salaries based on multiple factors.")

# Convert categorical inputs to numerical
def convert_inputs(position, qualification):
    position_map = {"lead generator": 1, "graphic designer": 2, "webdeveloper": 3, "Ai Engineer": 4, "Executive": 5}
    qual_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    return position_map[position], qual_map[qualification]

# Main function
def main():
    # Load models with error handling
    @st.cache_resource
    def load_models():
        try:
            model = safe_load_model("salary_model.h5")
            scaler = joblib.load("salary_scaler.pkl")
            return model, scaler
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None
    
    # Load data for visualization (optional)
    @st.cache_data
    def load_original_data():
        try:
            data = pd.read_csv("Salary_Data.csv")
            return data
        except:
            return None
    
    model, scaler = load_models()
    data = load_original_data()
    
    if model is not None and scaler is not None:
        # Make prediction when button is clicked
        if st.button("🔮 Predict Salary"):
            with st.spinner("Calculating prediction..."):
                try:
                    # Convert categorical inputs
                    position_num, qual_num = convert_inputs(position, qualification)
                    
                    # Prepare input array (MUST match training data feature order)
                    # Using only YearsExperience if scaler expects 1 feature
                    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == 1:
                        input_data = np.array([[years_exp]])
                        st.warning("Note: Using only YearsExperience for prediction (scaler expects 1 feature)")
                    else:
                        input_data = np.array([[years_exp, age, position_num, qual_num]])
                    
                    # Scale features
                    input_scaled = scaler.transform(input_data)
                    
                    # Predict
                    prediction = model.predict(input_scaled)[0][0]
                    
                    # Display results
                    st.success(f"Predicted salary: **${prediction:,.2f}**")
                    
                    # Visualization (if original data is available)
                    if data is not None:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot original data (using first feature for visualization)
                        ax.scatter(data.iloc[:, 0], data['Salary'], 
                                  color='blue', alpha=0.5, label='Historical Data')
                        
                        # Plot prediction
                        ax.scatter(years_exp, prediction, 
                                  color='red', s=200, label='Your Prediction')
                        
                        # Generate prediction trend line (using only YearsExperience)
                        x_values = np.linspace(0, 30, 100).reshape(-1, 1)
                        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 1:
                            # Create dummy inputs for other features if scaler expects multiple features
                            dummy_inputs = np.column_stack((
                                x_values,
                                np.full_like(x_values, age),
                                np.full_like(x_values, position_num),
                                np.full_like(x_values, qual_num)
                            ))
                            x_scaled = scaler.transform(dummy_inputs)
                        else:
                            x_scaled = scaler.transform(x_values)
                        
                        y_pred = model.predict(x_scaled)
                        ax.plot(x_values, y_pred, color='green', 
                               linewidth=2, label='Prediction Trend')
                        
                        ax.set_xlabel('Years of Experience')
                        ax.set_ylabel('Salary ($)')
                        ax.set_title('Salary Prediction')
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        
        # Model info section
        with st.expander("🧠 Model Information"):
            st.write("**Model Architecture:**")
            st.text("""
            Input Layer (1-4 neurons)
            ↓
            Hidden Layer 1 (64 neurons, ReLU activation)
            ↓
            Hidden Layer 2 (32 neurons, ReLU activation)
            ↓
            Output Layer (1 neuron, linear activation)
            """)
            
            st.write("**Input Features:**")
            if hasattr(scaler, 'n_features_in_'):
                st.write(f"Model expects {scaler.n_features_in_} features")
                if scaler.n_features_in_ == 1:
                    st.markdown("1. Years of Experience")
                else:
                    st.markdown("""
                    1. Years of Experience
                    2. Age
                    3. Position Level (1-5)
                    4. Qualification Level (1-4)
                    """)
            else:
                st.write("Feature information not available")

    else:
        st.warning("""
        Required files not found. Please ensure you have:
        - salary_model.h5 (Keras model)
        - salary_scaler.pkl (Scaler object)
        - Salary_Data.csv (Original data, optional)
        in the same directory as this app.
        """)

if __name__ == "__main__":
    main()