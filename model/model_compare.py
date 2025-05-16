import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns

# Load predictions
xgb_results = pd.read_csv('model/predictions_xgb.csv')
rf_results = pd.read_csv('model/predictions_rf.csv')

# Function to calculate metrics from predictions
def get_metrics(df):
    report = classification_report(df['actual'], df['predicted'], output_dict=True)
    return {
        'accuracy': (df['actual'] == df['predicted']).mean(),
        'precision_fake': report['1']['precision'],
        'recall_fake': report['1']['recall'],
        'f1_fake': report['1']['f1-score']
    }

# Get metrics
metrics_xgb = get_metrics(xgb_results)
metrics_rf = get_metrics(rf_results)

# Create DataFrame for plotting
comparison_df = pd.DataFrame([metrics_xgb, metrics_rf], index=['XGBoost', 'Random Forest'])

# Plot
plt.figure(figsize=(10,6))
comparison_df.plot(kind='bar', rot=0, colormap='viridis')
plt.title("Model Comparison (Fake Profile Detection)")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model/model_comparison.png')
plt.show()
