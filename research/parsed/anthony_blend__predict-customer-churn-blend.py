# %% cell 1
# Import dependency
from pathlib import Path
import pandas as pd

# %% cell 2
# Define the blending function
def blend_submissions(weight_dict, output_path, target_col):
    # Initialize list to store loaded DataFrames
    dataframes = []

    # Load each submission with its weight
    for path, weight in weight_dict.items():
        # Read the CSV file
        df = pd.read_csv(path)

        # Add a weighted prediction column
        df["weighted_pred"] = df[target_col] * weight

        # Append to list
        dataframes.append(df[["id", "weighted_pred"]])

    # Merge all submissions on 'id'
    merged = dataframes[0]
    for df in dataframes[1:]:
        # Merge on id
        merged = merged.merge(df, on="id", how="inner", suffixes=("", "_dup"))

        # Combine duplicate weighted_pred columns if any
        if "weighted_pred_dup" in merged.columns:
            merged["weighted_pred"] += merged["weighted_pred_dup"]
            merged.drop(columns=["weighted_pred_dup"], inplace=True)

    # Compute total weight
    total_weight = sum(weight_dict.values())

    # Compute blended prediction
    merged[target_col] = merged["weighted_pred"] / total_weight

    # Prepare final DataFrame
    blended = merged[["id", target_col]]

    # Save blended submission
    blended.to_csv(output_path, index=False)

    # Print confirmation
    print(f"✅ Blended submission saved to {output_path}")


# Define the main function
def main():
    # Target column
    TARGET = "Churn"

    # Define base input path
    BASE_PATH = Path("/kaggle/input/datasets/anthonytherrien/predict-customer-churn-vault/")

    # Define filenames with weights
    submission_files = [
        ("submission.csv", 2.7),
        ("submission (1).csv", 0.1),
    ]

    # Build full weight dictionary
    weight_dict = {
        str(BASE_PATH / filename): weight
        for filename, weight in submission_files
    }

    # Blend submissions
    blend_submissions(weight_dict, "submission.csv", TARGET)

# %% cell 3
# Call the main function
if __name__ == "__main__":
    main()

