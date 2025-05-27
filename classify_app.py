import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import category_encoders as ce
import gdown
import os

def download_if_not_exists(filename, gdrive_id):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        st.info(f"Downloading {filename} from Google Drive...")
        gdown.download(url, filename, quiet=False)



# Page configuration
st.set_page_config(
    page_title="Salary & Compensation Predictor",
    page_icon="💰",
    layout="wide"
)

# Load the trained model and preprocessors
@st.cache_resource
def load_model_components():
    import pickle

    def safe_pickle_load(filename, description):
        """Safely load a pickle file with error handling"""
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj, None
        except FileNotFoundError:
            return None, f"{description} file '{filename}' not found"
        except Exception as e:
            return None, f"Error loading {description}: {str(e)}"
    
    # 🧩 Step 1: Define GDrive file IDs
    gdrive_files = {
        "model_components.pkl": "1zkaI5gRiyYxS5w5lz9qdXCEVxrT9kk_v",
        "best_model.pkl": "1xFaGoQR-upcSlJIiPSt9AogQu9MLW_3h",
        "scaler.pkl": "1-vJlD7gF320W97_3QuiQMxt-UvzAYq_b",
        "pca.pkl": "18N9G_4EBHQd8XYy-WqsjAW3bDFnnBsoN",
        "target_encoder.pkl": "19hO81sSNi_Bd2CSpHXoILriaizOAX1GT",
        "ordinal_encoder.pkl": "1DQs0xqHkwuhudAlTZFXAX3APq-YGgfW6",
        "model_metadata.pkl": "1DKG6JqMYcPOwtsMZcnkZ8nFF3feLGqrv",
    }

    # 🧩 Step 2: Attempt to download each file if not present
    for file, gdrive_id in gdrive_files.items():
        download_if_not_exists(file, gdrive_id)

    # 🧩 Step 3: Load combined file if available
    combined_obj, combined_error = safe_pickle_load("model_components.pkl", "Combined components")
    if combined_obj is not None:
        return combined_obj
    
    st.info("Loading individual component files...")
    components = {}
    errors = []
    
    files_to_load = {
        'model': ("best_model.pkl", "Model"),
        'scaler': ("scaler.pkl", "Scaler"),
        'pca': ("pca.pkl", "PCA"),
        'target_encoder': ("target_encoder.pkl", "Target encoder"),
        'ordinal_encoder': ("ordinal_encoder.pkl", "Ordinal encoder"),
        'metadata': ("model_metadata.pkl", "Metadata"),
    }

    for key, (filename, desc) in files_to_load.items():
        obj, error = safe_pickle_load(filename, desc)
        if obj is not None:
            components[key] = obj
        else:
            errors.append(error)

    essential_components = ['model', 'scaler', 'pca', 'target_encoder']
    missing_essential = [comp for comp in essential_components if comp not in components]
    
    if missing_essential:
        st.error("Missing essential components:")
        for error in errors:
            st.error(f"• {error}")
        st.error("Please check that all required model files are accessible.")
        return None

    if errors:
        st.warning("Some non-essential components couldn't be loaded:")
        for error in errors:
            st.warning(f"• {error}")
    
    return components


def get_salary_tier_info():
    """Return information about what each salary tier represents"""
    # These are typical ranges based on common salary survey data
    # You should replace these with actual ranges from your dataset
    salary_tiers = {
        1: {"range": "$30,000 - $55,000", "description": "Entry Level"},
        2: {"range": "$55,000 - $75,000", "description": "Junior Level"}, 
        3: {"range": "$75,000 - $100,000", "description": "Mid Level"},
        4: {"range": "$100,000 - $140,000", "description": "Senior Level"},
        5: {"range": "$140,000+", "description": "Executive Level"}
    }
    
    additional_comp_tiers = {
        1: {"range": "$0 - $5,000", "description": "Minimal"},
        2: {"range": "$5,000 - $15,000", "description": "Low"},
        3: {"range": "$15,000 - $30,000", "description": "Moderate"},
        4: {"range": "$30,000 - $60,000", "description": "High"},
        5: {"range": "$60,000+", "description": "Very High"}
    }
    
    return salary_tiers, additional_comp_tiers

def display_tier_breakdown():
    """Display what each tier means in terms of salary ranges"""
    salary_tiers, additional_comp_tiers = get_salary_tier_info()
    
    st.subheader("📊 Tier Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Salary Tiers:**")
        for tier, info in salary_tiers.items():
            st.write(f"**Tier {tier} ({info['description']}):** {info['range']}")
    
    with col2:
        st.write("**Additional Compensation Tiers:**")
        for tier, info in additional_comp_tiers.items():
            st.write(f"**Tier {tier} ({info['description']}):** {info['range']}")
    
    st.info("💡 **Note:** These ranges are estimates based on typical market data. Actual ranges may vary based on your specific dataset.")

def main():
    st.title("💰 Salary & Additional Compensation Predictor")
    st.markdown("---")
    
    # Display tier breakdown
    display_tier_breakdown()
    
    st.markdown("---")
    
    # Load model and preprocessors
    components = load_model_components()
    if components is None:
        return
    
    model = components['model']
    scaler = components['scaler']
    pca = components['pca']
    target_encoder = components['target_encoder']
    
    # Get metadata if available
    metadata = components.get('metadata', {})
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        # Age selection
        age = st.selectbox(
            "Age Range",
            options=['under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 or over'],
            index=2
        )
        
        # Gender selection - FIXED to match training data
        gender = st.selectbox(
            "Gender",
            options=['Man', 'Woman', 'Non-binary', 'Other or prefer not to answer', 'Prefer not to answer'],
            index=0
        )
        
        # Race selection
        race = st.selectbox(
            "Race/Ethnicity",
            options=['White', 'Asian', 'Black or African American', 'Hispanic or Latino', 
                    'Mixed race', 'Native American', 'Other', 'Prefer not to answer'],
            index=0
        )
        
        # Education selection
        education = st.selectbox(
            "Education Level",
            options=['High School', 'Some college', 'College degree', "Master's degree",
                    'Professional degree (MD, JD, etc.)', 'PhD'],
            index=2
        )
    
    with col2:
        st.subheader("Professional Information")
        
        # Experience selection
        experience_overall = st.selectbox(
            "Overall Experience",
            options=['1 year or less', '2 - 4 years', '5-7 years', '8 - 10 years',
                    '11 - 20 years', '21 - 30 years', '31 - 40 years', '41 years or more'],
            index=3
        )
        
        experience_field = st.selectbox(
            "Experience in Current Field",
            options=['1 year or less', '2 - 4 years', '5-7 years', '8 - 10 years',
                    '11 - 20 years', '21 - 30 years', '31 - 40 years', '41 years or more'],
            index=2
        )
        
        # Industry selection
        industry = st.selectbox(
            "Industry",
            options=['Technology', 'Finance', 'Healthcare', 'Education', 'Government',
                    'Retail', 'Manufacturing', 'Consulting', 'Other'],
            index=0
        )
        
        # Job Title
        job_title = st.selectbox(
            "Job Title Category",
            options=['Software Engineer', 'Data Scientist', 'Manager', 'Analyst',
                    'Consultant', 'Developer', 'Director', 'Other'],
            index=0
        )
        
        # Country
        country = st.selectbox(
            "Country",
            options=['United States', 'Canada', 'United Kingdom', 'Germany',
                    'Australia', 'India', 'Other'],
            index=0
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Salary & Compensation", type="primary", use_container_width=True):
        
        try:
            # Create input dataframe with the same structure as training data
            input_data = pd.DataFrame({
                'Age': [age],
                'ExperienceOverall': [experience_overall],
                'ExperienceField': [experience_field],
                'Education': [education],
                'Industry_grouped': [industry],
                'JobTitle_grouped': [job_title],
                'Country_cleaned': [country],
                'Race': [race],
                'Gender': [gender]
            })
            
            # Apply the same preprocessing pipeline as training
            
            # 1. Target encoding for high cardinality features
            high_card_cols = ['Industry_grouped', 'JobTitle_grouped', 'Country_cleaned', 'Race']
            input_data[high_card_cols] = target_encoder.transform(input_data[high_card_cols])
            
            # 2. Ordinal encoding
            age_order = ['under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 or over']
            experience_order = ['1 year or less', '2 - 4 years', '5-7 years', '8 - 10 years',
                              '11 - 20 years', '21 - 30 years', '31 - 40 years', '41 years or more']
            education_order = ['High School', 'Some college', 'College degree', "Master's degree",
                             'Professional degree (MD, JD, etc.)', 'PhD']
            
            input_data['Age'] = age_order.index(age)
            input_data['ExperienceOverall'] = experience_order.index(experience_overall)
            input_data['ExperienceField'] = experience_order.index(experience_field)
            input_data['Education'] = education_order.index(education)
            
            # 3. One-hot encoding for Gender - FIXED to match training data
            input_data = pd.get_dummies(input_data, columns=['Gender'], prefix='Gender')
            
            # Ensure all expected gender columns are present (based on training data)
            expected_gender_cols = ['Gender_Man', 'Gender_Non-binary', 'Gender_Other or prefer not to answer', 
                                  'Gender_Prefer not to answer', 'Gender_Woman']
            for col in expected_gender_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Get the exact feature column order from metadata if available
            if metadata and 'feature_columns' in metadata:
                feature_cols = metadata['feature_columns']
                st.info(f"Using feature columns from metadata: {len(feature_cols)} features")
            else:
                # Fallback to manual ordering if metadata not available
                feature_cols = ['Age', 'ExperienceOverall', 'ExperienceField', 'Education',
                               'Industry_grouped', 'JobTitle_grouped', 'Country_cleaned', 'Race'] + expected_gender_cols
                st.warning("Using fallback feature ordering (metadata not found)")
            
            # Ensure we have all required columns
            missing_cols = [col for col in feature_cols if col not in input_data.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
                return
            
            # Select and reorder columns to match exact training order
            input_data = input_data[feature_cols]
            
            # 4. Apply scaling
            input_data_scaled = scaler.transform(input_data)
            
            # 5. Apply PCA
            input_data_pca = pca.transform(input_data_scaled)
            
            # 6. Make prediction
            prediction = model.predict(input_data_pca)
            
            # Display results
            st.success("Prediction Complete!")
            
            # Get tier information
            salary_tiers, additional_comp_tiers = get_salary_tier_info()
            
            predicted_salary_tier = int(prediction[0][0]) + 1
            predicted_comp_tier = int(prediction[0][1]) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Predicted Salary Range",
                    value=f"Tier {predicted_salary_tier}",
                    help="Salary tier from 1 (lowest) to 5 (highest)"
                )
                
                # Display the actual salary range and description
                if predicted_salary_tier in salary_tiers:
                    tier_info = salary_tiers[predicted_salary_tier]
                    st.write(f"**{tier_info['description']}**")
                    st.write(f"💰 **{tier_info['range']}**")
            
            with col2:
                st.metric(
                    label="Predicted Additional Compensation Range",
                    value=f"Tier {predicted_comp_tier}",
                    help="Additional compensation tier from 1 (lowest) to 5 (highest)"
                )
                
                # Display the actual additional comp range and description
                if predicted_comp_tier in additional_comp_tiers:
                    tier_info = additional_comp_tiers[predicted_comp_tier]
                    st.write(f"**{tier_info['description']}**")
                    st.write(f"🎁 **{tier_info['range']}**")
            
            # Calculate and display total compensation estimate
            st.markdown("---")
            
            # Extract numeric ranges for total calculation (simplified)
            salary_ranges = {
                1: (30000, 25000), 2: (52000, 70000), 3: (70000, 90000),
                4: (90000, 123000), 5: (123000, 1000000)
            }
            comp_ranges = {
                1: (0, 300000), 2: (300000, 600000), 3: (600000, 900000),
                4: (900000, 120000), 5: (120000, 150000)
            }
            
            salary_min, salary_max = salary_ranges.get(predicted_salary_tier, (0, 0))
            comp_min, comp_max = comp_ranges.get(predicted_comp_tier, (0, 0))
            
            total_min = salary_min + comp_min
            total_max = salary_max + comp_max
            
            st.subheader("💼 Total Compensation Estimate")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Minimum Total", f"${total_min:,}")
            with col2:
                st.metric("Maximum Total", f"${total_max:,}")
            with col3:
                st.metric("Average Estimate", f"${(total_min + total_max) // 2:,}")
            
            # Additional information
            st.info("""
            **Note:** 
            - Predictions are based on binned salary ranges (Tiers 1-5)
            - Tier 1 represents the lowest 20% of salaries in the dataset
            - Tier 5 represents the highest 20% of salaries in the dataset
            - Results are estimates based on the trained model and may not reflect exact salary amounts
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please check that the model was trained properly and all preprocessing components are saved.")
            st.info("Make sure to run the updated training script that saves all model components together.")
            
            # Debug information
            with st.expander("Debug Information"):
                st.write("Input data shape:", input_data.shape if 'input_data' in locals() else "Not created")
                st.write("Input data columns:", list(input_data.columns) if 'input_data' in locals() else "Not created")
                st.write("Expected feature columns:", feature_cols if 'feature_cols' in locals() else "Not defined")
                st.write("Metadata available:", bool(metadata))
                if metadata and 'feature_columns' in metadata:
                    st.write("Training feature columns:", metadata['feature_columns'])
                st.write("Missing columns:", missing_cols if 'missing_cols' in locals() else "None calculated")
    
    # Model information
    with st.expander("ℹ️ About This Model"):
        st.markdown("""
        This prediction model was trained using machine learning techniques including:
        - **Target Encoding** for high-cardinality categorical variables
        - **Ordinal Encoding** for ordered categorical variables
        - **Feature Scaling** using StandardScaler
        - **Dimensionality Reduction** using PCA
        - **Multi-output Classification** with ensemble methods (Random Forest, Gradient Boosting, XGBoost)
        
        The model predicts both salary and additional compensation ranges based on:
        - Personal demographics (age, gender, race, education)
        - Professional experience (overall and field-specific)
        - Job characteristics (industry, title, location)
        """)
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides estimates based on historical data and machine learning models. 
    Actual compensation may vary significantly based on company size, specific role requirements, 
    market conditions, and individual negotiation. Use these predictions as a general guide only.
    """)

if __name__ == "__main__":
    main()