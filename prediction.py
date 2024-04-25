import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xgb
import pickle

st.set_page_config(layout="wide")

# Initialize session state for page navigation if not already set
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# Define function to set the page
def set_page(page_name):
    st.session_state.page = page_name

# Apply CSS styling to ensure equal button widths with left-aligned text
st.markdown(
    """
    <style>
    div.stButton > button { /* Select all sidebar buttons */
        width: 200px;  /* Set consistent width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar buttons for page navigation
st.sidebar.button("Overview", on_click=set_page, args=("Overview",))
st.sidebar.button("Data Quality Analysis", on_click=set_page, args=("Data Quality Analysis",))
st.sidebar.button("Binary Classification", on_click=set_page, args=("Binary Classification",))
st.sidebar.button("Multiclass Classification", on_click=set_page, args=("Multiclass Classification",))
#----------------------------------------------------------------------------------------------------

if st.session_state.page == "Overview":
    st.markdown("""
    <style>
    .title {
        margin-top: -90px;
        font-size: 40px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="title">An IoT Network Intrusion Detection and Classification with XGBoost using CICIOT2023 Dataset</h1>', unsafe_allow_html=True)

    # Binary and Multiclass Classification Data Setup
    binary_metrics = ['Precision', 'Recall', 'F1 Score']
    binary_classifiers = ['Model 1', 'Model 2', 'Model 3']
    binary_scores = np.array([
        [0.9959, 0.9951, 0.9954],  # Model 1 (All Features)
        [0.9959, 0.9951, 0.9953],  # Model 2 (RF)
        [0.9959, 0.9950, 0.9953]   # Model 3 (ET)
   
    ])

    multiclass_metrics = ['Precision', 'Recall', 'F1 Score']
    multiclass_classifiers = ['Model 1', 'Model 2', 'Model 3']
    multiclass_scores = np.array([
        [0.9963, 0.9953, 0.9957],  # Model 1 (All Features)
        [0.9961, 0.9950, 0.9954],  # Model 2 (RF)
        [0.9958, 0.9947, 0.9951]   # Model 3 (ET)
    ])

    # Define color palettes
    binary_color_palette = ['#FA7070', '#A1C398', '#C6EBC5']
    multiclass_color_palette = ['#891652', '#EABE6C', '#FFEDD8']

    # Adding a spacer using markdown
    #st.markdown('###')

    # Define custom text for each model
    binary_model_labels = ['All Features', 'Top 25 RF Feature Importance', 'Top 25 ET Feature Importance']
    multiclass_model_labels = binary_model_labels  # Assuming the same labeling for simplicity

    # Splitting the screen into two columns for binary and multiclass comparisons
    col1, mid_col, col2 = st.columns([1,1,1])

    # Feature Selection
    with col1:
        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px; margin-bottom: -15px;'>Introduction</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: justify">
        The proliferation of Internet of Things (IoT) technology has surged in recent years, promising transformative impacts across various industries. However, this growth is accompanied by a concerning rise in cyberattacks targeting IoT devices, as evidenced by the comprehensive CICIoT2023 dataset (Neto et al., 2023). To address this pressing issue, the project focuses on developing and evaluating an IoT network intrusion detection and classification system using state-of-the-art machine learning techniques. Leveraging the CICIoT2023 dataset, supervised learning methods are employed, with a particular emphasis on the highly effective XGBoost algorithm known for its capability to handle imbalanced datasets. Furthermore, three feature selection approaches are explored to enhance the XGBoost model's performance in both binary and multiclass classifications of IoT network intrusions as following:
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px; margin-bottom: -30px;'>Feature Selection Methodologies:</h1>", unsafe_allow_html=True)

        st.markdown("""
            <style>
                .bordered-box {
                    border: 2px solid #ff4b4b; /* Adjust the color as needed */
                    padding: 5px;
                    border-radius: 5px; /* Rounded corners */
                    margin: 5px 0px; /* Reduced vertical margin for spacing between boxes */
                    color: #ffffff; /* Text color */
                    background-color: #000000; /* Adjust background color as needed */
                }
                .bordered-box p {
                    margin-bottom: 0px; /* Reduce bottom margin of the paragraph */
                }
                .first-box {
                    margin-top: -15px; /* Remove top margin from the first box */
                }
                .model-header {
                    color: #ff4b4b; /* Header text color */
                    font-weight: bold;
                }
            </style>
            """, unsafe_allow_html=True)

        # Model 1
        st.markdown("""
                    <div class="bordered-box">
                        <span class="model-header">Model 1:</span>
                        <p>Using all features</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Model 2
        st.markdown("""
                    <div class="bordered-box">
                        <span class="model-header">Model 2:</span>
                        <p>Feature Selection using Top 25 Random Forest (RF) Feature Importance</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Model 3
        st.markdown("""
                    <div class="bordered-box">
                        <span class="model-header">Model 3:</span>
                        <p>Feature Selection using Top 25 Extra Trees (ET) Feature Importance</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Plotting Binary Classification Model Performance
    with mid_col:      
        # Display the performance comparison and key insights text
        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px;  margin-bottom: -40px;'>Binary Model Performance Analysis</h1>", unsafe_allow_html=True)
        # Custom CSS for text alignment
        st.markdown(
            """
            <style>
            .justify {
                text-align: justify;
            }
            .highlighted {
                color: #ff4b4b;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="justify">
                <ul style="list-style-type: none; padding-left: 0; margin-left: 0;">
                    <li class="highlighted"><strong>F1-Score:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>The F1-Scores are closely aligned across all models, indicating a balanced trade-off between precision and recall with minimal variations.</li>
                    </ul>
                    <li class="highlighted"><strong>Recall:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>Strong recall rates across all models, with negligible differences, suggesting high detection rates for positive cases.</li>
                    </ul>
                    <li class="highlighted"><strong>Precision:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>All three models have high precision, indicating that most positive classifications are correct.</li>
                    </ul>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px; margin-top: -20px; margin-bottom: -30px;'>Key Insights</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="justify">
            <ul>
                <li>Models 2 and 3, which use reduced feature sets, achieve comparable performance to Model 1, which uses all features.</li>
                <li>This suggests that feature reduction can improve computational efficiency without significantly affecting model accuracy.</li>
                <li>Minor variations in performance among the models are likely due to different methods used for feature selection.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        binary_fig = go.Figure()
        for idx, classifier in enumerate(binary_classifiers):
            hover_text = f"{binary_model_labels[idx]}<br>Score: %{{x:.4f}}"  # Custom hover text
            binary_fig.add_trace(go.Bar(
                x=binary_scores[idx],
                y=binary_metrics,
                name=classifier,
                marker_color=binary_color_palette[idx],
                hoverinfo="text",  # Use custom text for hover
                hovertemplate=hover_text,  # Set hover text
                orientation='h'
            ))
        # Update layout configuration
        binary_fig.update_layout(
            title=dict(text='Binary Model Performance Comparison', x=0.2, y=1),
            legend=dict(orientation='h', x=0.2, y=1, xanchor='center', yanchor='bottom'),
            barmode='group',
            xaxis=dict(showticklabels=False, range=[0.9945, 0.9965]),
            margin=dict(t=45, l=150)
        )
        st.plotly_chart(binary_fig, use_container_width=True)

    # Plotting Multiclass Classification Model Performance
    with col2:
        # Display the performance comparison and key insights text
        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px;  margin-bottom: -40px;'>Multiclass Model Performance Analysis</h1>", unsafe_allow_html=True)
        # Custom CSS for text alignment
        st.markdown(
            """
            <style>
            .justify {
                text-align: justify;
            }
            .highlighted {
                color: #ff4b4b;
            }
            </style>
            """
        , unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="justify">
                <ul style="list-style-type: none; padding-left: 0; margin-left: 0;">
                    <li class="highlighted"><strong>F1-Score:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>Model 1 has the highest F1 Score, suggesting a strong balance between precision and recall.</li>
                        <li>Models 2 and 3 have slightly lower F1 Scores, indicating some trade-offs in balancing precision and recall.</li>
                    </ul>
                    <li class="highlighted"><strong>Recall:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>Model 1 has the highest recall, indicating that it identifies the most positive cases among the three models.</li>
                        <li>Models 2 and 3 have progressively lower recall, implying that they might miss more positive cases compared to Model 1.</li>
                    </ul>
                    <li class="highlighted"><strong>Precision:</strong></li>
                    <ul style="list-style-type: disc; padding-left: 15px; margin-left: -15px;">
                        <li>Model 1 has the highest precision, indicating that when it predicts a positive case, it is most likely correct.</li>
                        <li>Model 2 has slightly lower precision, while Model 3 has the lowest precision among the three.</li>
                    </ul>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 16px; margin-top: -20px; margin-bottom: -30px;'>Key Insights</h1>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="justify">
            <ul>
                <li>Model 1 consistently outperforms Models 2 and 3 across all metrics, indicating that using all features might yield better overall results in a multiclass setting.</li>
                <li>Models 2 and 3, which use feature selection methods, tend to have lower recall and precision, suggesting that the reduced feature sets might lead to missing some positive cases or producing more false positives.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        multiclass_fig = go.Figure()
        for idx, classifier in enumerate(multiclass_classifiers):
            hover_text = f"{multiclass_model_labels[idx]}<br>Score: %{{x:.4f}}"  # Custom hover text
            multiclass_fig.add_trace(go.Bar(
                x=multiclass_scores[idx],
                y=multiclass_metrics,
                name=classifier,
                marker_color=multiclass_color_palette[idx],
                hoverinfo="text",  # Use custom text for hover
                hovertemplate=hover_text,  # Set hover text
                orientation='h'
            ))
        # Update layout configuration
        multiclass_fig.update_layout(
            title=dict(text='Multiclass Model Performance Comparison', x=0.2, y=1),
            legend=dict(orientation='h', x=0.2, y=1, xanchor='center', yanchor='bottom'),
            barmode='group',
            xaxis=dict(showticklabels=False, range=[0.9940,0.9966]),
            margin=dict(t=45, l=150)
        )
        st.plotly_chart(multiclass_fig, use_container_width=True)

#----------------------------------------------------------------------------------------------------------------------------

if st.session_state.page == "Data Quality Analysis":
    st.title('Data Quality Analysis')


    
#----------------------------------------------------------------------------------------------------------------------------

# CSS to create border for metrics
st.markdown("""
<style>
.metric-box {
  border: 2px solid #ffffff; /* White border */
  border-radius: 5px; /* Rounded corners */
  padding: 10px;
  margin-bottom: 40px; /* Adds space between boxes */
  margin-left: 40px; /* Adjusts space on the left, moving the boxes to the right */
  display: flex;
  flex-direction: column;
  justify-content: center; /* Centers content vertically */
  align-items: center; /* Centers content horizontally */
  height: 100px; /* Fixed height */
  width: 200px; /* Fixed width */
  background-color: #FF5349; 
}
.metric-title {
  color: #ffffff; /* White text color */
  font-size: 18px; /* Adjust title font size */
  margin: 0;
}
.metric-value {
  color: #ffffff; /* White text color */
  font-size: 25px; /* Adjust value font size */
  font-weight: bold;  
  margin: 0;
}          
.hyper-title {
    font-size: 20px; /* Title font size */
    font-weight: bold;
    margin: 0; /* Space below the title */
}
.hyper-param-name, .hyper-param-value {
    display: block;
    text-align: left;
    margin-bottom: 3px;
}
.hyper-param-name {
    font-size: 18px; /* Adjust as needed */
    margin-top: 10px; /* Space above each hyperparameter name */
}
.hyper-param-value {
    font-size: 20px; /* Adjust as needed */
    font-weight: bold;
    color: #000000; /* Black font color */
    padding: 1px; /* Add padding as needed */
    background-color: #D3D3D3; /* Light grey background color */
    display: inline-block; /* Allows padding to take effect */
    width: calc(100% - 16px); /* Full width minus padding */
    box-sizing: border-box; /* Include padding in the width calculation */
    margin-bottom: 10px; /* Space below the value if needed */
}
</style>
""", unsafe_allow_html=True)

# Function to display metrics with styled borders, fixed sizes, and adjusted font sizes
def display_metrics(metrics):
    for metric, value in metrics.items():
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{metric}</div>
                <div class="metric-value">{value:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

if st.session_state.page == "Binary Classification":
    st.title('Binary Classification Model')

    # Model metrics as dictionaries for ease of use
    model_metrics = {
        'Model 1: All Features': {
            'Precision': 0.9959,
            'Recall': 0.9951,
            'F1-Score': 0.9954
        },
        'Model 2: Top 25 using RF Feature Importance': {
            'Precision': 0.9959,
            'Recall': 0.9951,
            'F1-Score': 0.9953
        },
        'Model 3: Top 25 using ET Feature Importance': {
            'Precision': 0.9959,
            'Recall': 0.9950,
            'F1-Score': 0.9953
        }
    }

    # Model hyperparameters for display
    model_hyperparams = {
        'Model 1: All Features': {
            'scale_pos_weight': 42.5532,
            'reg_alpha': 0.5,
            'reg_lambda': 0,
            'random_state': 42
        },
        'Model 2: Top 25 using RF Feature Importance': {
            'scale_pos_weight': 42.5532,
            'reg_alpha': 0.9,
            'reg_lambda': 0.3,
            'random_state': 42
        },
        'Model 3: Top 25 using ET Feature Importance': {
            'scale_pos_weight': 42.5532,
            'reg_alpha': 0.3,
            'reg_lambda': 0,
            'random_state': 42
        }
    }

    # Function to display hyperparameters with styled CSS
    def display_hyperparams(hyperparams):
        st.markdown("<div class='hyper-box'>", unsafe_allow_html=True)
        st.markdown("<div class='hyper-title'>Model Hyperparameters:</div>", unsafe_allow_html=True)
        for param, value in hyperparams.items():
            st.markdown(f"""
                <div>
                    <span class='hyper-param-name'>{param.replace('_', '_')}:</span>
                    <span class='hyper-param-value'>{value}</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # Create tabs for different feature sets
    tab_all_features, tab_rf_25, tab_et_25 = st.tabs([
        "Model 1: All Features", 
        "Model 2: Top 25 using RF Feature Importance", 
        "Model 3: Top 25 using ET Feature Importance"
    ])

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    def preprocess_and_predict(data, scaler_path, model_path, model_type):
        # Drop specific columns based on the model type
        if model_type == 'RF':
            columns_to_drop = ['fin_count', 'ack_count', 'HTTP', 'psh_flag_number', 'UDP',
                               'syn_flag_number', 'rst_flag_number', 'ICMP', 'SSH', 'DNS',
                               'fin_flag_number', 'LLC', 'IPv', 'ARP', 'ece_flag_number',
                               'cwr_flag_number', 'DHCP', 'IRC', 'Drate', 'Telnet', 'SMTP']
            data = data.drop(columns=columns_to_drop, errors='ignore')
        elif model_type == 'ET':
            columns_to_drop = ['TCP', 'Covariance', 'rst_flag_number', 'ack_count', 'fin_count',
                               'UDP', 'HTTP', 'ICMP', 'SSH', 'fin_flag_number', 'DNS', 'IPv',
                               'LLC', 'ARP', 'ece_flag_number', 'cwr_flag_number', 'DHCP',
                               'Drate', 'IRC', 'Telnet', 'SMTP']
            data = data.drop(columns=columns_to_drop, errors='ignore')

        
        # Load the model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler_loaded = pickle.load(f)

        # Preprocess data
        feature_columns = [col for col in data.columns if col not in ['label', 'Binary Class', 'Multiclass']]
        X = data[feature_columns]
        X_scaled = scaler_loaded.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)
        labels = ['attack', 'benign']
        predicted_labels = [labels[pred] for pred in predictions]

        # Create DataFrame
        df_with_predictions = pd.DataFrame(X_scaled, columns=feature_columns)
        df_with_predictions['Predicted Label'] = predicted_labels

        return df_with_predictions

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        with tab_all_features:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 1: All Features'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler.pkl', 'binary_allfeatures.json', 'All Features')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="all_features_filter")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 1: All Features'])

        with tab_rf_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 2: Top 25 using RF Feature Importance'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_rf.pkl', 'binary_rf_top25.json', 'RF')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="rf_25_filter")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 2: Top 25 using RF Feature Importance'])


        with tab_et_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 3: Top 25 using ET Feature Importance'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_et.pkl', 'binary_et_top25.json', 'ET')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="et_25_filter")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 3: Top 25 using ET Feature Importance'])

#--------------------------------------------------------------------------------------------------------------------------------------------------

if st.session_state.page == "Multiclass Classification":
    st.title('Multiclass Classification Model')

    # Model metrics as dictionaries for ease of use
    model_metrics = {
        'Model 1: All Features': {
            'Precision': 0.9963,
            'Recall': 0.9953,
            'F1-Score': 0.9957
        },
        'Model 2: Top 25 using RF Feature Importance': {
            'Precision': 0.9961,
            'Recall': 0.9950,
            'F1-Score': 0.9954
        },
        'Model 3: Top 25 using ET Feature Importance': {
            'Precision': 0.9958,
            'Recall': 0.9947,
            'F1-Score': 0.9951
        }
    }

    # Model hyperparameters for display
    model_hyperparams = {
        'Model 1: All Features': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.5,
            'reg_lambda': 0.7,
            'random_state': 42
        },
        'Model 2: Top 25 using RF Feature Importance': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.3,
            'reg_lambda': 0.6,
            'random_state': 42
        },
        'Model 3: Top 25 using ET Feature Importance': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'random_state': 42
        }
    }

    # Create tabs for different feature sets
    tab_all_features, tab_rf_25, tab_et_25 = st.tabs([
        "Model 1: All Features", 
        "Model 2: Top 25 using RF Feature Importance", 
        "Model 3: Top 25 using ET Feature Importance"
    ])

    # Function to display hyperparameters with styled CSS
    def display_hyperparams(hyperparams):
        st.markdown("<div class='hyper-box'>", unsafe_allow_html=True)
        st.markdown("<div class='hyper-title'>Model Hyperparameters:</div>", unsafe_allow_html=True)
        for param, value in hyperparams.items():
            st.markdown(f"""
                <div>
                    <span class='hyper-param-name'>{param.replace('_', '_')}:</span>
                    <span class='hyper-param-value'>{value}</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    def preprocess_and_predict(data, scaler_path, model_path, model_type):
        # Drop specific columns based on the model type
        if model_type == 'RF':
            columns_to_drop = ['TCP', 'fin_count', 'ack_count', 'psh_flag_number', 'HTTPS',
                               'syn_flag_number', 'fin_flag_number', 'rst_flag_number', 'HTTP',
                               'SSH', 'DNS', 'LLC', 'IPv', 'ARP', 'ece_flag_number', 'Drate',
                               'cwr_flag_number', 'DHCP', 'IRC', 'Telnet', 'SMTP']
            data = data.drop(columns=columns_to_drop, errors='ignore')
        elif model_type == 'ET':
            columns_to_drop = ['rst_count', 'rst_flag_number', 'Std', 'Radius', 'HTTPS',
                               'urg_count', 'ack_count', 'Covariance', 'HTTP', 'SSH', 'DNS',
                               'LLC', 'IPv', 'ARP', 'Drate', 'ece_flag_number', 'DHCP',
                               'cwr_flag_number', 'SMTP', 'IRC', 'Telnet']
            data = data.drop(columns=columns_to_drop, errors='ignore')

        # Load the model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Load the scaler
        with open(scaler_path, 'rb') as f:
            scaler_loaded = pickle.load(f)

        # Preprocess data
        feature_columns = [col for col in data.columns if col not in ['label', 'Binary Class', 'Multiclass']]
        X = data[feature_columns]
        X_scaled = scaler_loaded.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)
        labels = ['benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
        predicted_labels = [labels[pred] for pred in predictions]

        # Create DataFrame
        df_with_predictions = pd.DataFrame(X_scaled, columns=feature_columns)
        df_with_predictions['Predicted Label'] = predicted_labels

        return df_with_predictions

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        with tab_all_features:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 1: All Features'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_multi_all.pkl', 'multi_all.json', 'All Features')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="allfeatures")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 1: All Features'])

        with tab_rf_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 2: Top 25 using RF Feature Importance'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_multi_rf.pkl', 'multi_rf.json', 'RF')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="rf_25_multi")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 2: Top 25 using RF Feature Importance'])


        with tab_et_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 3: Top 25 using ET Feature Importance'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_multi_et.pkl', 'multi_et.json', 'ET')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="et_25_multi")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=300)
            with col2:
                st.markdown("""
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                """, unsafe_allow_html=True)
                display_metrics(model_metrics['Model 3: Top 25 using ET Feature Importance'])

#----------------------------------------------------------------------------------------------------------------------------------------------------------


