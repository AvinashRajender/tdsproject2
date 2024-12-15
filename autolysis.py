# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "scikit-learn",
#   "tabulate",
# ]
# ///


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import chardet
import json
import subprocess
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Ensure AIPROXY_TOKEN is set
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")
# Function to read CSV with automatic encoding detection
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

def read_csv(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)
# Function for summary statistics
def summary_statistics(df):
    summary = df.describe(include='all')
    print("Summary Statistics:")
    print(summary)
    return summary
# Function to count missing values
def missing_values(df):
    missing = df.isnull().sum()
    print("Missing Values:")
    print(missing)
    return missing
# Function to calculate correlation matrix
def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    print("Correlation Matrix:")
    print(correlation)
    return correlation

# Function to detect outliers
def detect_outliers(df):
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return pd.Series([], dtype=int)
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)  # Parallelize with n_jobs=-1
    numeric_data["outliers"] = iso.fit_predict(numeric_data)
    print("Outliers detected:")
    print(numeric_data["outliers"].value_counts())
    return numeric_data["outliers"]


# Function to perform clustering analysis
def clustering_analysis(df):
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if numeric_data.empty:
        return pd.Series([], dtype=int, name="Cluster")
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Reduce n_init to speed up
    clusters = kmeans.fit_predict(numeric_data)
    df['Cluster'] = pd.Series(clusters, index=numeric_data.index)
    
    print("Clustering Results:")
    print(df['Cluster'].value_counts())
    
    return df['Cluster']



# Visualization functions
def plot_correlation_matrix(correlation, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm")
    plt.savefig(output_file)
    plt.close()

def plot_pairplot(df, output_file):
    plt.figure(figsize=(10, 8))
    sns.pairplot(df.select_dtypes(include=[np.number]).dropna().sample(n=min(100, len(df))), plot_kws={'alpha':0.5})
    plt.savefig(output_file)
    plt.close()



def plot_clusters(df, output_file):
    if 'Cluster' not in df.columns:
        print("No clusters found to plot.")
        return
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['Cluster'], palette='viridis')
    plt.savefig(output_file)
    plt.close()



def query_llm(prompt):
    """
    Queries the LLM for insights and returns the response.
    """
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",  # Supported chat model
            "messages": [
                {"role": "system", "content": "You are a helpful data analysis assistant. Provide insights, suggestions, and implications based on the given analysis and visualizations."},
                {"role": "user", "content": prompt},
            ],
        }
        payload_json = json.dumps(payload)
        curl_command = [
            "curl",
            "-X", "POST", url,
            "-H", f"Authorization: Bearer {AIPROXY_TOKEN}",
            "-H", "Content-Type: application/json",
            "-d", payload_json
        ]
        result = subprocess.run(curl_command, capture_output=True, text=True)
        if result.returncode == 0:
            response_data = json.loads(result.stdout)
            return response_data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error in curl request: {result.stderr}")
    except Exception as e:
        print(f"Error querying AI Proxy: {e}")
        return "Error: Unable to generate narrative."
def create_story(analysis, visualizations_summary):
    """
    Creates a narrative using LLM based on analysis and visualizations.
    """
    prompt = (
        f"### Data Summary:\nShape: {analysis['shape']}\n"
        f"Columns: {', '.join(list(analysis['columns'].keys()))}\n"
        f"Missing Values: {str(analysis['missing_values'])}\n\n"
        f"### Key Summary Statistics:\nTop 3 Columns:\n{pd.DataFrame(analysis['summary_statistics']).iloc[:, :3].to_string()}\n\n"
        f"### Visualizations:\nCorrelation heatmap, Pairplot, Clustering Scatter Plot.\n\n"
        "Based on the above, provide a detailed narrative including insights and potential actions."
    )

    return query_llm(prompt)
def save_results(analysis, visualizations, story, output_folder):
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Data Overview\n")
        f.write(f"**Shape**: {analysis['shape']}\n\n")
        f.write("## Summary Statistics\n")
        f.write(pd.DataFrame(analysis["summary_statistics"]).to_markdown())
        f.write("\n\n## Narrative\n")
        f.write(str(story))  # If story is a list of strings, join them

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = dataset_name

    print(f"Reading file: {file_path}")
    print(f"Output folder created: {output_folder}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Detecting file encoding...")
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    df = read_csv(file_path)
    print("Dataframe loaded.")

    # Perform analysis
    analysis = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include="all").to_dict(),
    }
    correlation = correlation_matrix(df)
    outliers = detect_outliers(df)
    clusters = clustering_analysis(df)

    # Generate visualizations
    plot_correlation_matrix(correlation, os.path.join(output_folder, "correlation_heatmap.png"))
    plot_pairplot(df, os.path.join(output_folder, "pairplot.png"))
    plot_clusters(df, os.path.join(output_folder, "clustering_scatter.png"))

    # Generate story using LLM
    visualizations_summary = "Correlation heatmap, Pairplot, Clustering Scatter Plot."
    story = create_story(analysis, visualizations_summary)

    # Save results
    save_results(analysis, visualizations_summary, story, output_folder)

    print("Analysis complete.")
    print(f"Generated visualizations: {['correlation_heatmap.png', 'pairplot.png', 'clustering_scatter.png']}")
    print("Story created.")
    print("Results saved.")

if __name__ == "__main__":
    main()
