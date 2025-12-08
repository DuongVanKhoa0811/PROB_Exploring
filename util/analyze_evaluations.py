import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

def parse_combined_evaluation_file(file_path: str) -> List[Dict[str, float]]:
    """
    Parse a single log file containing multiple evaluation runs.
    
    Args:
        file_path: Path to the combined evaluation log file
        
    Returns:
        List of dictionaries, each containing metrics from one evaluation run
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define patterns to extract each metric
    patterns = {
        'Current_class_AP50': r'Current class AP50: tensor\(([\d.]+)\)',
        'Current_class_Precisions50': r'Current class Precisions50: ([\d.]+)',
        'Current_class_Recall50': r'Current class Recall50: ([\d.]+)',
        'Known_AP50': r'Known AP50: tensor\(([\d.]+)\)',
        'Known_Precisions50': r'Known Precisions50: ([\d.]+)',
        'Known_Recall50': r'Known Recall50: ([\d.]+)',
        'Unknown_AP50': r'Unknown AP50: tensor\(([\d.]+)\)',
        'Unknown_Precisions50': r'Unknown Precisions50: ([\d.]+)',
        'Unknown_Recall50': r'Unknown Recall50: ([\d.]+)',
    }
    
    # First, find how many evaluation runs exist
    # Use one metric as reference to count runs
    test_pattern = patterns['Current_class_AP50']
    num_runs = len(re.findall(test_pattern, content))
    
    print(f"Found {num_runs} evaluation runs in {Path(file_path).name}")
    
    # Extract all values for each metric using findall
    all_metrics_values = {}
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, content)
        all_metrics_values[metric_name] = [float(m) for m in matches]
        
        if len(matches) != num_runs:
            print(f"Warning: {metric_name} has {len(matches)} values, expected {num_runs}")
    
    # Convert to list of dictionaries (one dict per run)
    metrics_list = []
    for i in range(num_runs):
        run_metrics = {'run_number': i + 1}
        for metric_name in patterns.keys():
            if i < len(all_metrics_values[metric_name]):
                run_metrics[metric_name] = all_metrics_values[metric_name][i]
            else:
                raise ValueError(f"Metric {metric_name} has {len(all_metrics_values[metric_name])} values, expected {num_runs}")
        metrics_list.append(run_metrics)
    
    return metrics_list

def analyze_combined_evaluation_file(file_path: str, output_csv: str = 'evaluation_summary.csv'):
    """
    Analyze a single log file with multiple evaluation runs and export statistics.
    
    Args:
        file_path: Path to the combined evaluation log file
        output_csv: Path to output CSV file
    """
    print(f"Processing: {file_path}")
    print("="*80)
    
    # Parse the combined file
    try:
        all_runs = parse_combined_evaluation_file(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
    
    # Create DataFrame with all runs
    df = pd.DataFrame(all_runs)
    
    # Calculate statistics (excluding 'run_number' column)
    metric_columns = [col for col in df.columns if col != 'run_number']
    
    # Create summary statistics
    summary_data = []
    
    for metric in metric_columns:
        values = df[metric].dropna()
        if len(values) > 0:
            mean = values.mean()
            std = values.std()
            min_val = values.min()
            max_val = values.max()
            summary_data.append({
                'Metric': metric,
                'Mean': mean,
                'Std': std,
                'Min': min_val,
                'Max': max_val,
                'Mean_±_Std': f"{mean:.2f} ± {std:.2f}",
                'Count': len(values)
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"\nSummary statistics saved to: {output_csv}")
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Also save detailed results (all runs)
    detailed_csv = output_csv.replace('.csv', '_detailed.csv')
    df.to_csv(detailed_csv, index=False)
    print(f"\nDetailed results (all runs) saved to: {detailed_csv}")
    
    # Display detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS (All Runs)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return summary_df, df

# Example usage:
if __name__ == "__main__":
    # Single combined log file
    combined_log_file = '../logs/logs_PROB_MOWODB_EVAL_V17_1.txt'

    if not Path(combined_log_file).exists():
        print(f"Error: File not found: {combined_log_file}")
        print("Please update the file path.")
    else:
        summary_df, detailed_df = analyze_combined_evaluation_file(
            combined_log_file,
            output_csv='evaluation_summary_V17_1.csv'
        )