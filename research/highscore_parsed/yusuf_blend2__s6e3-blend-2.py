# %% cell 1
import pandas as pd

# Define the blending function
def blend_submissions(weight_dict, output_path):
    # Initialize list to store loaded DataFrames
    dataframes = []

    # Load each submission with its weight
    for path, weight in weight_dict.items():
        # Read the CSV file
        df = pd.read_csv(path)

        # Add a weighted prediction column
        df["weighted_pred"] = df["Churn"] * weight

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
    merged["Churn"] = merged["weighted_pred"] / total_weight

    # Prepare final DataFrame
    blended = merged[["id", "Churn"]]

    # Save blended submission
    blended.to_csv(output_path, index=False)

    # Print confirmation
    print(f"Blended submission saved to {output_path}")
# Define the main function
def main():
    # Define file paths and their respective weights
    weight_dict = {
        "/kaggle/input/notebooks/nina2025/ps-s6e3-hb15/submission.csv": 2.7,
        "/kaggle/input/notebooks/yusufmurtaza01/s6e3-blending/submission.csv": 0.1,
    }
    # Call blend function
    blend_submissions(weight_dict, output_path="submission.csv")

# Call the main function
if __name__ == "__main__":
    main()

