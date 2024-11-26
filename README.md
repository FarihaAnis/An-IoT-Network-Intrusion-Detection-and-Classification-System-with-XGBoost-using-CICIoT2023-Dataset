# IoT Network Intrusion Detection System

## Objective
To develop a machine learning-based intrusion detection and classification system for IoT networks using XGBoost to enhance cybersecurity measures against a wide spectrum of network attacks.

## Tools and Libraries
- **Data Manipulation**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn
- **Data Preprocessing**: LabelEncoder, RobustScaler
- **Feature Selection**: Random Forest, Extra Trees
- **Model Development**:
  - **Classification Models**: XGBoost (Binary and Multiclass)
- **Model Evaluation**: accuracy_score, f1_score, precision_score, recall_score
- **Dashboard Development**: Streamlit
- **Utility**: sklearn.utils.class_weight

## Dataset
The CICIoT2023 dataset from the Canadian Institute for Cybersecurity provides a comprehensive simulation of IoT attack scenarios. It comprises:
- **Records**: 7,332,065 samples selected from an original dataset of 46,686,748 records.
- **Attributes**: 47 features, focusing on seven attack classes: DDoS, DoS, Recon, Web-Based, Brute Force, Spoofing, Mirai, and a ‘Benign’ class.

## Methodology
1. **Data Preprocessing**:
   - Cleaned and encoded the dataset using LabelEncoder.
   - Applied RobustScaler to normalize data.
   - Addressed class imbalance using `scale_pos_weight` for binary classification and balanced class weights for multiclass classification.
2. **Feature Selection**:
   - Extracted top 25 features using Random Forest and Extra Trees based on feature importance scores to optimize model performance.
3. **Model Development**:
   - Trained XGBoost models for binary and multiclass classification tasks.
   - Conducted hyperparameter tuning, including adjustments to class weights and L1/L2 regularization parameters.
4. **Dashboard Integration**:
   - Developed an interactive Streamlit dashboard to showcase model predictions on new datasets.

## Models

### Performance Metrics

| **Classification Type** | **Model**                | **Accuracy** | **F1-Score** | **Precision** | **Recall** |
|--------------------------|--------------------------|--------------|--------------|---------------|------------|
| **Binary**               | XGBoost with Random Forest | 99.87%       | 99.85%       | 99.88%        | 99.84%     |
| **Binary**               | XGBoost with Extra Trees  | 99.84%       | 99.83%       | 99.86%        | 99.82%     |
| **Multiclass**           | XGBoost with Random Forest | 98.5%        | 98.3%        | 98.4%         | 98.2%      |
| **Multiclass**           | XGBoost with Extra Trees  | 98.3%        | 98.1%        | 98.2%         | 98.0%      |

The integration of XGBoost with both Random Forest and Extra Trees demonstrates outstanding performance in both binary and multiclass scenarios. The combination with Random Forest slightly outperforms Extra Trees across most metrics, particularly in binary classification.

## Validation
- Performed evaluation using unseen data comprising 446,796 records, ensuring robust and generalizable performance.
- Incorporated cross-validation to mitigate overfitting.

## Conclusion
The integration of XGBoost with Random Forest and Extra Trees delivers exceptional performance, achieving over 99% in all key metrics for binary classification and over 98% for multiclass classification. Both combinations prove highly effective in detecting IoT network attacks and handling imbalanced class distributions. This robust framework ensures reliable and accurate intrusion detection, making it an excellent choice for enhancing IoT network security.
