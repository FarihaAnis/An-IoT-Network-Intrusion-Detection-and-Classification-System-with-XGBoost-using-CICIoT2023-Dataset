import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import pickle
from PIL import Image


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

st.sidebar.markdown("<h1 style='text-align: left; color: #ffffff; font-size: 24px;'>Navigation Panel: </h1>", unsafe_allow_html=True)
# Sidebar buttons for page navigation
st.sidebar.button("Overview", on_click=set_page, args=("Overview",))
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
          
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 24px; margin-left: -30px;'>Increasing IoT Devices Worldwide Drive Cybersecurity Threat Surge</h1>", unsafe_allow_html=True)
        # Data for the plot
        year = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
        connected_devices = [8.6, 9.76, 11.28, 13.14, 15.14, 17.08, 19.08, 21.09, 23.14, 25.21, 27.31, 29.42]

        # Find the index where forecast starts
        forecast_start_index = year.index(2025)

        # Create the figure
        fig = go.Figure()
        # Add historical data line with area shading
        fig.add_trace(go.Scatter(
            x=year[:forecast_start_index],
            y=connected_devices[:forecast_start_index],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#ffba08', width=4),
            marker=dict(symbol='cross', color='#ffba08', size=8),
            fill='tozeroy'  # Fill to zero on y-axis
        ))

        # Add shadow for historical data (by adding slightly offset areas with decreasing opacity)
        for i in range(1, 4):  # You can adjust the range for more/less shadow layers
            fig.add_trace(go.Scatter(
                x=year[:forecast_start_index],
                y=[y - i * 0.2 for y in connected_devices[:forecast_start_index]],  # Shift down each layer
                mode='lines',
                line=dict(width=0),
                fill='tonexty',  # Fill to the next trace
                fillcolor=f'rgba(157, 2, 8, {max(0.1, 0.5 - i * 0.1)})',
                showlegend=False
            ))

        # Add forecast data line with area shading
        fig.add_trace(go.Scatter(
            x=year[forecast_start_index-1:],  # Start from the last historical point
            y=connected_devices[forecast_start_index-1:],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#9d0208', width=4),
            marker=dict(symbol='cross', color='#9d0208', size=8),
            fill='tozeroy'
        ))

        # Add shadow for forecast data
        for i in range(1, 4):  # Similar shadow effect for the forecast
            fig.add_trace(go.Scatter(
                x=year[forecast_start_index-1:],
                y=[y - i * 0.2 for y in connected_devices[forecast_start_index-1:]],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba(255, 186, 8, {max(0.1, 0.5 - i * 0.1)})',
                showlegend=False
            ))

        # Add annotations for the y-values at 2009 and 2030
        fig.add_annotation(
            x=year[0],  # Index for year 2009
            y=connected_devices[1],
            text=str(connected_devices[1]) + "Billions",  # Display the value with 'B' for billion
            showarrow=True,
            arrowhead=1,
            arrowcolor='white',
            ax=30,
            ay=-30
        )

        fig.add_annotation(
            x=year[-1],  # Index for year 2030
            y=connected_devices[-1],
            text=str(connected_devices[-1]) + "Billions",  # Display the value with 'B' for billion
            showarrow=True,
            arrowhead=1,
            arrowcolor='white',
            ax=-30,
            ay=-30
        )
        # Update layout with titles and axis labels
        fig.update_layout(
            xaxis_title='Year',
            xaxis=dict(tickmode='linear', dtick=1, range=[2018.5, 2030.5]),
            yaxis=dict(showticklabels=False),
            legend=dict(
                x=0.25,
                y=1.15,
                xanchor='center',
                yanchor='top',
                orientation='h'
            ),
            template='plotly_white',
            margin=dict(l=10,t=20,r=100),
            width=630,
            height=630
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
            <div style='position: relative; top: -50px; font-weight: normal; font-size: 9px;'>
                Source: Vailshery, L. S. (2023, July 27). IoT connected devices worldwide 2019-2030. Statista. 
                <a href='https://www.statista.com/statistics/1183457/iot-connected-devices-worldwide/'>Retrieved from Statista</a>. 
            </div>
            """, unsafe_allow_html=True)
        
    with col1:
        st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 25px; margin-top: -30px;'>The Importance of Attack Classification</h1>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top: 20px;'> </div>", unsafe_allow_html=True)
        st.image('classification.png')

    with col2:
        st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 25px;'>Why Choose CICIoT2023?</h1>", unsafe_allow_html=True)
        # Split col2 into two new columns
        col2a, col2b = st.columns(2)
        with col2a:
            st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 16px; margin-top: -25px;'>Extensive and Realistic Network Topology</h1>", unsafe_allow_html=True)
            st.image('topology.png')
            st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 16px; margin-top: -10px;'>Data from Actual IoT Devices</h1>", unsafe_allow_html=True)
            st.image('iotdevices.png')

        with col2b:
            # Title
            st.markdown("""
                <h1 style='text-align: center; color: white; font-size: 16px; margin-top: -25px;'>
                    Distribution of Diverse Communication Protocols
                </h1>
            """, unsafe_allow_html=True)

            # Data Preparation
            data = {
                "Protocol": ["HTTP", "HTTP", "HTTPS", "HTTPS", "DNS", "DNS", "Telnet", "Telnet", "SMTP", "SMTP", "SSH", "SSH", "IRC", "IRC", "TCP", "TCP", "UDP", "UDP", "DHCP", "DHCP", "ARP", "ARP", "ICMP", "ICMP", "IPv", "IPv", "LLC", "LLC"],
                "Class": ["Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign", "Attack", "Benign"],
                "Value": [4.836619e-02, 3.868120e-02, 3.943823e-02, 7.096014e-01, 8.939268e-05, 1.865131e-03, 0.000000e+00, 0.000000e+00, 1.396761e-07, 0.000000e+00, 4.581375e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 5.666690e-01, 8.601789e-01, 2.155660e-01, 6.818155e-02, 2.095141e-06, 0.000000e+00, 6.089876e-05, 2.954090e-04, 1.676906e-01, 5.792333e-06, 9.999025e-01, 9.992760e-01, 9.999025e-01, 9.992760e-01]
            }
            df = pd.DataFrame(data)

            # Colors for classes
            colors = {'Attack': '#9d0208', 'Benign': '#ffba08'}

            # Create bar plot using plotly.graph_objects with switched axes
            fig = go.Figure()

            for i, class_name in enumerate(df['Class'].unique()):
                class_data = df[df['Class'] == class_name]
                fig.add_trace(go.Bar(y=class_data['Protocol'], x=class_data['Value'], name=class_name, orientation='h', marker_color=colors[class_name]))

            # Update layout to move legend to the top and remove x-axis name
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=1.2, xanchor="center", x=0.18, title=None),
                            xaxis=dict(title="Distribution"),
                            #title=dict(text='Protocol', x=0.3, y=1),
                            height=290,
                            width=400,
                            margin=dict(t=0,b=20))

            # Streamlit Display
            st.plotly_chart(fig, use_container_width=True)

#########################################################
            # Data for the plot
            count = ['72.81%', '17.31%', '5.64%', '2.35%', '1.05%', '0.76%', '0.05%', '0.03%']
            attack = ['DDoS', 'DoS', 'Mirai', 'Benign', 'Spoofing', 'Recon', 'Web', 'BruteForce']

            count.reverse()
            attack.reverse()

            # Convert percentage strings to float for plotting
            percentage = [float(x.strip('%')) for x in count]

            # Define your custom color palette
            colors = ['#ffba08', '#faa307', '#f48c06', '#e85d04', '#dc2f02', '#d00000', '#9d0208', '#6a040f']

            # Initialize the figure
            multiclass_fig = go.Figure()

            # Add bars
            for idx, pct in enumerate(percentage):
                hover_text = f"{count[idx]}"
                multiclass_fig.add_trace(go.Bar(
                    x=[pct],  # Use percentage for y-axis
                    y=[attack[idx]],  # Use attack type for x-axis
                    name=attack[idx],
                    marker_color=colors[idx],  # Ensure each attack type gets a unique color
                    hoverinfo="text",
                    hovertemplate=hover_text,
                    orientation='h'
                ))

            # Update layout configuration to remove y-axis ticks and horizontal grid lines
            multiclass_fig.update_layout(
                title=dict(text='Diverse Attack Simulation', x=0.3, y=1),
                barmode='stack',  # Use stack to better represent percentages
                xaxis=dict(
                    range=[0, max(percentage) * 1.1],  # Set range to slightly above the highest percentage
                    tickfont=dict(size=16),
                    titlefont=dict(size=16),
                    showticklabels=False,  # Hide y-axis tick labels
                    showgrid=False,  # Hide y-axis grid lines
                    zeroline=False  # Hide the zero line
                ),
                yaxis=dict(
                    tickfont=dict(size=16),
                    titlefont=dict(size=16)
                ),
                showlegend=False,
                margin=dict(t=30,l=100),  # Adjust margins to fit legend and title
                template='plotly_white',
                height=350,  # Adjust the height (in pixels)
                width=400   # Adjust the width (in pixels)
            )

            # Display the figure in Streamlit
            st.plotly_chart(multiclass_fig, use_container_width=True)

    with col2:
        st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 25px; margin-top: -50px;'>Efficient Classification with XGBoost</h1>", unsafe_allow_html=True)
        col2c, col2d = st.columns(2)
        with col2c:
            st.markdown("""
                        <h1 style='text-align: center; color: white; font-size: 16px; margin-top: -25px;'>
                        Effective Handling of Imbalanced Datasets
                        </h1>
                        """, unsafe_allow_html=True)   
            # Data for binary classification
            binary_algorithms = ['XGBoost', 'Logistic Regression*', 'AdaBoost*', 'Random Forest*']
            binary_f1_scores = [0.9954, 0.8763, 0.9563, 0.9653]
            binary_algorithms.reverse()
            binary_f1_scores.reverse()

            # Data for multiclass classification
            multiclass_algorithms = ['XGBoost', 'Logistic Regression*', 'AdaBoost*', 'Random Forest*']
            multiclass_f1_scores = [0.9957, 0.5394, 0.3687, 0.7193]
            multiclass_algorithms.reverse()
            multiclass_f1_scores.reverse()

            # Create figure
            fig = go.Figure()

            # Add binary classification data
            fig.add_trace(go.Bar(
                y=binary_algorithms,
                x=binary_f1_scores,
                name='Binary Classification',
                orientation='h',
                marker_color='#ffba08',
                text=[f'<b>F1-Score: {x:.4f}</b>' for x in binary_f1_scores],  # Format text labels with 4 decimal places
                textposition='inside',  # Position text inside the bars
                insidetextanchor='middle'  # Center text horizontally inside the bars
            ))

            # Add multiclass classification data
            fig.add_trace(go.Bar(
                y=multiclass_algorithms,
                x=multiclass_f1_scores,
                name='Multiclass Classification',
                orientation='h',
                marker_color='#9d0208',
                text=[f'<b>F1-Score: {x:.4f}</b>' for x in multiclass_f1_scores],  # Format text labels with 4 decimal places
                textposition='inside',  # Position text inside the bars
                insidetextanchor='middle'  # Center text horizontally inside the bars
            ))

            # Update layout to remove any extra space and set font size for readability
            fig.update_layout(
                barmode='group',
                legend=dict(orientation="h", 
                            yanchor="top", 
                            y=1.1, 
                            xanchor="left", 
                            x=-0.1, 
                            title=None),
                margin=dict(t=0, b=0),
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                xaxis=dict(
                    showticklabels=False,  # Hide x-axis tick labels
                    showgrid=False,        # Optionally, remove the grid lines
                    ticks=''               # Hide x-axis ticks
                ),
                height=350,
                width=400           
            )

            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
                <div style='position: relative; top: -30px; font-weight: normal; font-size: 9px;'>
                    *Benchmark:Neto, E., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.
                        </div>
                """, unsafe_allow_html=True)

        with col2d:
            st.markdown("""
                        <h1 style='text-align: center; color: white; font-size: 16px; margin-top: -25px;'>
                        Workflow with Weight Distribution Adjustment and Regularization
                        </h1>
                        """, unsafe_allow_html=True)    
            st.image('xgbworkflow.png')          
   
#----------------------------------------------------------------------------------------------------------------------------

# CSS to create border for metrics
st.markdown("""
<style>       
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

if st.session_state.page == "Binary Classification":
    st.title('Binary Classification Model')
    st.markdown("""
    - **Purpose**: To demonstrate and assess XGBoost binary classification models performance with new, unseen datasets.
    - **Feature Comparison**: Uses full feature set, top 25 Random Forest (RF), and top 25 Extra Trees (ET) features.
    - **User Interaction**: Supports uploading of CSV files for dynamic testing.
    """)

    # Model hyperparameters for display
    model_hyperparams = {
        'Model 1: All Features': {
            'scale_pos_weight': 42.5532,
            'reg_alpha': 0.5,
            'reg_lambda': 0,
            'random_state': 42
        },
        'Model 2: Top 25 using RF Feature Importance Ranking': {
            'scale_pos_weight': 42.5532,
            'reg_alpha': 0.9,
            'reg_lambda': 0.3,
            'random_state': 42
        },
        'Model 3: Top 25 using ET Feature Importance Ranking': {
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
        "Model 2: Top 25 using RF Feature Importance Ranking", 
        "Model 3: Top 25 using ET Feature Importance Ranking"
    ])

    #uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

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

    # Check if file uploaded
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        # Store the uploaded file data in session state
        st.session_state.uploaded_file_data = pd.read_csv(uploaded_file)

    # Use the uploaded file data wherever needed
    if st.session_state.uploaded_file_data is not None:
        data = st.session_state.uploaded_file_data

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
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px; margin-top:-20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=150)
                
                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:
                    # Define new colors for specific labels and values for the x-axis
                    label_colors = {'benign': '#6d6d6d', 'attack': '#dbdbdb'}
                    label_values = {'benign': 10420, 'attack': 436375}

                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total_count = sum(label_values.values())

                    # Assign the colors based on the new label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total_count) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title, remove x and y axis ticks, and apply custom colors
                    fig.update_layout(
                        title=dict(
                            text='Benchmark',
                            xanchor='center',
                            x=0.5,  # Center the title
                            y=1
                        ),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=0.97,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)



                with mid_col2b:
                    # Define colors for specific labels
                    label_colors = {'attack': '#d00000', 'benign': '#ffba08'}

                    # Count the occurrences of each label and prepare data for the bar chart
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_index(ascending=False)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the sum of counts for percentage calculations
                    total = sum(counts)

                    # Assign colors based on label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout for the bar chart with a legend at the top
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='top',
                            y=1.25,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False,  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False,  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        template='plotly_white',
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.59,
                    "Recall": 99.51,
                    "F1-Score": 99.54
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       
       
        with tab_rf_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 2: Top 25 using RF Feature Importance Ranking'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_rf.pkl', 'binary_rf_top25.json', 'RF')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="rf_25_filter")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px; margin-top:-20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=150)

                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:
                    # Define new colors for specific labels and values for the x-axis
                    label_colors = {'benign': '#6d6d6d', 'attack': '#dbdbdb'}
                    label_values = {'benign': 10420, 'attack': 436375}

                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total_count = sum(label_values.values())

                    # Assign the colors based on the new label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total_count) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title, remove x and y axis ticks, and apply custom colors
                    fig.update_layout(
                        title=dict(
                            text='Benchmark',
                            xanchor='center',
                            x=0.5,  # Center the title
                            y=1
                        ),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=0.97,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)



                with mid_col2b:
                    # Define colors for specific labels
                    label_colors = {'attack': '#d00000', 'benign': '#ffba08'}

                    # Count the occurrences of each label and prepare data for the bar chart
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_index(ascending=False)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the sum of counts for percentage calculations
                    total = sum(counts)

                    # Assign colors based on label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout for the bar chart with a legend at the top
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='top',
                            y=1.25,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False,  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False,  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        template='plotly_white',
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.59,
                    "Recall": 99.51,
                    "F1-Score": 99.53
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       
        with tab_et_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 3: Top 25 using ET Feature Importance Ranking'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_et.pkl', 'binary_et_top25.json', 'ET')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="et_25_filter")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px; margin-top:-20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=150)

                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:
                    # Define new colors for specific labels and values for the x-axis
                    label_colors = {'benign': '#6d6d6d', 'attack': '#dbdbdb'}
                    label_values = {'benign': 10420, 'attack': 436375}

                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total_count = sum(label_values.values())

                    # Assign the colors based on the new label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total_count) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title, remove x and y axis ticks, and apply custom colors
                    fig.update_layout(
                        title=dict(
                            text='Benchmark',
                            xanchor='center',
                            x=0.5,  # Center the title
                            y=1
                        ),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom',
                            y=0.97,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                with mid_col2b:
                    # Define colors for specific labels
                    label_colors = {'attack': '#d00000', 'benign': '#ffba08'}

                    # Count the occurrences of each label and prepare data for the bar chart
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_index(ascending=False)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the sum of counts for percentage calculations
                    total = sum(counts)

                    # Assign colors based on label names
                    bar_colors = [label_colors.get(label, '#888888') for label in labels]

                    # Create a separate trace for each label to generate a legend
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{(counts[i] / total) * 100:.2f}%</b>"],  # Display as percentages with two decimal places
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],
                            name=labels[i],  # Use the label name as the trace name
                            showlegend=True,
                            hoverinfo='skip'
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout for the bar chart with a legend at the top
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=20,
                        uniformtext_mode='hide',
                        legend=dict(
                            orientation='h',
                            yanchor='top',
                            y=1.25,
                            xanchor='center',
                            x=0.5
                        ),
                        xaxis=dict(
                            showticklabels=False,  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False,  # Remove y-axis ticks
                        ),
                        margin=dict(t=55),
                        template='plotly_white',
                        height=250
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.59,
                    "Recall": 99.50,
                    "F1-Score": 99.53
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       
#--------------------------------------------------------------------------------------------------------------------------------------------------

if st.session_state.page == "Multiclass Classification":
    st.title('Multiclass Classification Model')
    st.markdown("""
    - **Purpose**: To demonstrate and assess XGBoost multiclass classification models performance with new, unseen datasets.
    - **Feature Comparison**: Uses full feature set, top 25 RF, and top 25 ET features.
    - **User Interaction**: Supports uploading of CSV files for dynamic testing.
    """)
    # Model hyperparameters for display
    model_hyperparams = {
        'Model 1: All Features': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.5,
            'reg_lambda': 0.7,
            'random_state': 42
        },
        'Model 2: Top 25 using RF Feature Importance Ranking': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.3,
            'reg_lambda': 0.6,
            'random_state': 42
        },
        'Model 3: Top 25 using ET Feature Importance Ranking': {
            'class_weights': [5.3116, 460.5568, 0.1717, 0.7222, 2.2127, 16.5379, 11.8497, 236.6491],
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'random_state': 42
        }
    }

    # Create tabs for different feature sets
    tab_all_features, tab_rf_25, tab_et_25 = st.tabs([
        "Model 1: All Features", 
        "Model 2: Top 25 using RF Feature Importance Ranking", 
        "Model 3: Top 25 using ET Feature Importance Ranking"
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

    #uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

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

    # Check if file uploaded
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None

    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        # Store the uploaded file data in session state
        st.session_state.uploaded_file_data = pd.read_csv(uploaded_file)

    # Use the uploaded file data wherever needed
    if st.session_state.uploaded_file_data is not None:
        data = st.session_state.uploaded_file_data

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
                st.dataframe(filtered_df, height=150)

                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:

                    label_values = {
                        'BruteForce': 104,
                        'Web': 219,
                        'Recon': 3287,
                        'Spoofing': 4659,
                        'benign': 10420,
                        'Mirai': 25166,
                        'DoS': 77317,
                        'DDoS': 325623,
                    }                    

                    # New color palette matching each label
                    label_colors = {
                        'DDoS': '#e8e8e8',
                        'DoS': '#d0d0d0',
                        'Mirai': '#c0c0c0',
                        'benign': '#b0b0b0',
                        'Spoofing': '#a0a0a0',
                        'Recon': '#909090',
                        'Web': '#808080',
                        'BruteForce': '#505050'
                    }

                    
                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total = sum(label_values.values())

                    # Assign the colors based on the specified labels
                    bar_colors = [label_colors[label] for label in labels]

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color for each bar
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Benchmark', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=30),
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


                with mid_col2b:
                    # Count the occurrences of each label in the filtered DataFrame and sort them in descending order
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_values(ascending=True)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the total count for percentage calculations
                    total = sum(counts)

                    # New color palette
                    multiclass_colors = ['#ffba08', '#faa307', '#f48c06', '#e85d04', '#dc2f02', '#d00000', '#9d0208', '#6a040f']
                    bar_colors = multiclass_colors[:len(labels)]  # Ensure the color list matches the number of labels

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color from the multiclass palette
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=30),
                        template='plotly_white',
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.63,
                    "Recall": 99.53,
                    "F1-Score": 99.57
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       


        with tab_rf_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 2: Top 25 using RF Feature Importance Ranking'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_multi_rf.pkl', 'multi_rf.json', 'RF')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="rf_25_multi")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=150)

                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:

                    label_values = {
                        'BruteForce': 104,
                        'Web': 219,
                        'Recon': 3287,
                        'Spoofing': 4659,
                        'benign': 10420,
                        'Mirai': 25166,
                        'DoS': 77317,
                        'DDoS': 325623,
                    }                    

                    # New color palette matching each label
                    label_colors = {
                        'DDoS': '#e8e8e8',
                        'DoS': '#d0d0d0',
                        'Mirai': '#c0c0c0',
                        'benign': '#b0b0b0',
                        'Spoofing': '#a0a0a0',
                        'Recon': '#909090',
                        'Web': '#808080',
                        'BruteForce': '#505050'
                    }
                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total = sum(label_values.values())

                    # Assign the colors based on the specified labels
                    bar_colors = [label_colors[label] for label in labels]

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color for each bar
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Benchmark', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=30),
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


                with mid_col2b:
                    # Count the occurrences of each label in the filtered DataFrame and sort them in descending order
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_values(ascending=True)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the total count for percentage calculations
                    total = sum(counts)

                    # New color palette
                    multiclass_colors = ['#ffba08', '#faa307', '#f48c06', '#e85d04', '#dc2f02', '#d00000', '#9d0208', '#6a040f']
                    bar_colors = multiclass_colors[:len(labels)]  # Ensure the color list matches the number of labels

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color from the multiclass palette
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        template='plotly_white',
                        margin=dict(t=30),
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.61,
                    "Recall": 99.50,
                    "F1-Score": 99.54
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       

        with tab_et_25:
            col1, mid_col, col2 = st.columns([1, 3, 1])
            with col1:
                display_hyperparams(model_hyperparams['Model 3: Top 25 using ET Feature Importance Ranking'])
            with mid_col:
                df_predictions = preprocess_and_predict(data, 'robust_scaler_multi_et.pkl', 'multi_et.json', 'ET')
                # Multi-select filter for "Predicted Label"
                label_options = df_predictions['Predicted Label'].unique().tolist()
                selected_labels = st.multiselect("Filter by Predicted Label", label_options, default=label_options,  key="et_25_multi")
                # Filter the DataFrame based on the selected labels
                filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
                # Display the filtered DataFrame
                st.markdown("<h1 style='text-align: center; color: #ffffff; font-size: 20px;'>Prediction Results:</h1>", unsafe_allow_html=True)
                st.dataframe(filtered_df, height=150)

                mid_col2a, mid_col2b = st.columns([2,2])
                with mid_col2a:

                    label_values = {
                        'BruteForce': 104,
                        'Web': 219,
                        'Recon': 3287,
                        'Spoofing': 4659,
                        'benign': 10420,
                        'Mirai': 25166,
                        'DoS': 77317,
                        'DDoS': 325623,
                    }                    

                    # New color palette matching each label
                    label_colors = {
                        'DDoS': '#e8e8e8',
                        'DoS': '#d0d0d0',
                        'Mirai': '#c0c0c0',
                        'benign': '#b0b0b0',
                        'Spoofing': '#a0a0a0',
                        'Recon': '#909090',
                        'Web': '#808080',
                        'BruteForce': '#505050'
                    }
                    # Extract the labels and counts based on the selected labels
                    filtered_label_values = {label: count for label, count in label_values.items() if label in selected_labels}

                    # Extract the filtered labels and counts to lists for plotting
                    labels = list(filtered_label_values.keys())
                    counts = list(filtered_label_values.values())

                    # Calculate the total count for percentage calculations
                    total = sum(label_values.values())

                    # Assign the colors based on the specified labels
                    bar_colors = [label_colors[label] for label in labels]

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color for each bar
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Benchmark', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        margin=dict(t=30),
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


                with mid_col2b:
                    # Count the occurrences of each label in the filtered DataFrame and sort them in descending order
                    label_counts = filtered_df['Predicted Label'].value_counts().sort_values(ascending=True)
                    labels = label_counts.index.tolist()
                    counts = label_counts.values.tolist()

                    # Calculate the total count for percentage calculations
                    total = sum(counts)

                    # New color palette
                    multiclass_colors = ['#ffba08', '#faa307', '#f48c06', '#e85d04', '#dc2f02', '#d00000', '#9d0208', '#6a040f']
                    bar_colors = multiclass_colors[:len(labels)]  # Ensure the color list matches the number of labels

                    # Create a separate trace for each label to generate the bar chart
                    data_traces = [
                        go.Bar(
                            x=[counts[i]],  # Single bar for each category
                            y=[labels[i]],
                            text=[f"<b>{labels[i]}: {(counts[i] / total) * 100:.2f}%</b>"],  # Add label name with percentage
                            textposition='auto',
                            orientation='h',
                            marker_color=bar_colors[i],  # Assign the color from the multiclass palette
                            showlegend=False,  # Disable legend for individual traces
                            hoverinfo='skip'  # Suppress hover text
                        )
                        for i in range(len(labels))
                    ]

                    # Create the bar chart using plotly.graph_objects (go) with separate traces
                    fig = go.Figure(data=data_traces)

                    # Update layout to center the title and remove x and y axis ticks
                    fig.update_layout(
                        title=dict(text='Distribution of Predicted Labels', x=0.5, xanchor='center', y=1),
                        uniformtext_minsize=12,
                        uniformtext_mode='hide',
                        xaxis=dict(
                            showticklabels=False  # Remove x-axis ticks
                        ),
                        yaxis=dict(
                            showticklabels=False  # Remove y-axis ticks
                        ),
                        template='plotly_white',
                        margin=dict(t=30),
                        height=350
                    )

                    # Display the Plotly bar chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Add a centered title
                st.markdown(
                    """
                    <h1 style="text-align: center; color: #ffffff; font-size: 20px;">Performance Metrics:</h1>
                    """, unsafe_allow_html=True
                )

                # Function to display a donut chart with metric names using Plotly
                def display_donut_chart(metric_title, value, max_value=100, color='#FF5349'):
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[metric_title, ""],
                            values=[value, max_value - value],
                            hole=.7,
                            marker_colors=[color, 'lightgray'],
                            textinfo='none'
                        )
                    ])
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        annotations=[
                            {
                                "text": f"<b>{metric_title}</b>",
                                "x": 0.5, "y": 0.6,  # Position above the value
                                "font_size": 12,
                                "showarrow": False,
                                "font": {"color": "white"}
                            },
                            {
                                "text": f"<b>{value:.2f}</b>",
                                "x": 0.5, "y": 0.45,  # Position in the center of the chart
                                "font_size": 14,
                                "showarrow": False,
                                "font": {"color": "#FFB466"}
                            }
                        ],
                        height=110, width=150
                    )
                    st.plotly_chart(fig)

                # Your provided metrics in a dictionary
                metrics = {
                    "Precision": 99.58,
                    "Recall": 99.47,
                    "F1-Score": 99.51
                }

                # Create a centered layout
                for metric, value in metrics.items():
                    # Create an empty space to center the chart
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col2:
                        display_donut_chart(metric, value, max_value=100, color='#FF5349')
       
