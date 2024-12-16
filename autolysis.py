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
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
import chardet
import json
import subprocess
import re
import hashlib
from scipy.stats import ttest_ind, f_oneway
from dateutil import parser
import matplotlib 
matplotlib.use('Agg')

# Ensure AIPROXY_TOKEN is set
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise EnvironmentError("AIPROXY_TOKEN is not set. Please set it before running the script.")

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

# Function to parse date strings using regex
def parse_date_with_regex(date_str):
    if not isinstance(date_str, str):  
        return date_str

    if not re.search(r'\d', date_str):  # Check for digits in the string
        return np.nan

    patterns = [
        (r"\d{2}-[A-Za-z]{3}-\d{4}", "%d-%b-%Y"), 
        (r"\d{2}-[A-Za-z]{3}-\d{2}", "%d-%b-%y"),
        (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
        (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),
        (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y")
    ]

    for pattern, date_format in patterns:
        if re.match(pattern, date_str):
            try:
                return pd.to_datetime(date_str, format=date_format, errors='coerce')
            except Exception as e:
                print(f"Error parsing date: {date_str} with format {date_format}. Error: {e}")
                return np.nan

    try:
        return parser.parse(date_str, fuzzy=True, dayfirst=False)
    except Exception as e:
        print(f"Error parsing date with dateutil: {date_str}. Error: {e}")
        return np.nan

# Function to detect date columns
def detect_date_column(column):
    if isinstance(column, str):
        if any(keyword in column.lower() for keyword in ['date', 'time', 'timestamp']):
            return True

    sample_values = column.dropna().head(10)
    date_patterns = [r"\d{2}-[A-Za-z]{3}-\d{2}", r"\d{2}-[A-Za-z]{3}-\d{4}", r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}"]

    for value in sample_values:
        if isinstance(value, str):
            for pattern in date_patterns:
                if re.match(pattern, value):
                    return True
    return False

# Function to read CSV with encoding detection and date parsing
def read_csv(file_path):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')
        for column in df.columns:
            if df[column].dtype == object and detect_date_column(df[column]):
                df[column] = df[column].apply(parse_date_with_regex)
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
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    numeric_data["outliers"] = iso.fit_predict(numeric_data)
    print("Outliers detected:")
    print(numeric_data["outliers"].value_counts())
    return numeric_data["outliers"]

# Function to perform clustering analysis
def clustering_analysis(df):
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    if numeric_data.empty:
        return pd.Series([], dtype=int, name="Cluster")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(numeric_data)
    df['Cluster'] = pd.Series(clusters, index=numeric_data.index)

    print("Clustering Results:")
    print(df['Cluster'].value_counts())

    return df['Cluster']

# Function to perform statistical tests
def statistical_tests(df):
    numeric_data = df.select_dtypes(include=[np.number])
    results = {}

    for col1 in numeric_data.columns:
        for col2 in numeric_data.columns:
            if col1 != col2:
                stat, p_value = ttest_ind(numeric_data[col1].dropna(), numeric_data[col2].dropna())
                results[f"T-test: {col1} vs {col2}"] = {"Statistic": stat, "P-value": p_value}

    if numeric_data.shape[1] > 2:
        anova_stat, anova_p_value = f_oneway(*[numeric_data[col].dropna() for col in numeric_data.columns])
        results["ANOVA"] = {"Statistic": anova_stat, "P-value": anova_p_value}

    return results

# Function to perform regression analysis

def regression_analysis(df):
    numeric_data = df.select_dtypes(include=[np.number])
    if numeric_data.shape[1] < 2:
        return None

    # Handle missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    numeric_data_imputed = imputer.fit_transform(numeric_data)

    x = numeric_data_imputed[:, :-1]
    y = numeric_data_imputed[:, -1]

    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    feature_importance = dict(zip(numeric_data.columns[:-1], np.abs(model.coef_)))

    return {
        "MSE": mean_squared_error(y, predictions),
        "R2": r2_score(y, predictions),
        "Feature Importance": feature_importance,
    }


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

# Function to query LLM for insights
def query_llm(prompt):
    try:
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
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

# Function to cache LLM query results
def cache_llm_query(function_call, cache_dir="llm_cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Hash the function_call to create a unique key
    query_hash = hashlib.md5(function_call.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{query_hash}.json")

    # Check if cache exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    # If not cached, query the LLM
    response = query_llm(function_call)
    with open(cache_file, "w") as f:
        json.dump(response, f)

    return response

# Function to create a narrative using LLM based on analysis and visualizations
def create_story(analysis, visualizations_summary):
    prompt = (
        f"### Data Summary:\nShape: {analysis['shape']}\n"
        f"Columns: {', '.join(list(analysis['columns'].keys()))}\n"
        f"Missing Values: {str(analysis['missing_values'])}\n\n"
        f"### Key Summary Statistics:\nTop 3 Columns:\n{pd.DataFrame(analysis['summary_statistics']).iloc[:, :3].to_string()}\n\n"
        f"### Visualizations:\nCorrelation heatmap, Pairplot, Clustering Scatter Plot.\n\n"
        "Based on the above, provide a detailed narrative including insights and potential actions."
    )

    return cache_llm_query(prompt)

# Function to save results in README.md
def save_results(analysis, visualizations, story, output_folder):
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Data Overview\n")
        f.write(f"**Shape**: {analysis['shape']}\n\n")
        f.write("## Summary Statistics\n")
        f.write(pd.DataFrame(analysis["summary_statistics"]).to_markdown())
        f.write("\n\n## Visualizations\n")
        f.write("Below are the key visualizations generated during the analysis:\n\n")
        for vis in visualizations:
            vis_filename = os.path.basename(vis)
            if "correlation_heatmap" in vis_filename:
                explanation = "The heatmap highlights the correlations between numerical features. Strong correlations may indicate predictive relationships."
            elif "clustering_scatter" in vis_filename:
                explanation = "The scatter plot shows clustering patterns, which can help identify natural groupings in the data."
            elif "pairplot" in vis_filename:
                explanation = "The pairplot provides pairwise visualizations of feature relationships, which are useful for identifying trends and dependencies."
            else:
                explanation = "This visualization provides additional insights into the dataset."

            f.write(f"- **{explanation}**\n  ![Visualization]({vis_filename})\n\n")
        f.write("\n\n## Narrative\n")
        f.write("### Key Insights and Narrative\n\n")
        f.write(story)
        f.write("\n\n## Conclusions and Recommendations\n\n")
        f.write("The analysis revealed significant trends and patterns that are critical for understanding the dataset. Recommendations for further exploration and potential action items are outlined below:\n\n")
        f.write("- Address missing data through imputation or collection improvements.\n")
        f.write("- Focus on high-correlation features for predictive modeling.\n")
        f.write("- Investigate outliers to understand their context and impact.\n")
        f.write("- Use clustering insights for targeted interventions or segmentation.\n")

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
    analysis["statistical_tests"] = statistical_tests(df)

    # Perform regression analysis
    regression_results = regression_analysis(df)
    if regression_results:
        analysis["regression_analysis"] = regression_results

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
