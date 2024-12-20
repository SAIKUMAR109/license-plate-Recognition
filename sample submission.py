import pandas as pd

# Load and inspect the SampleSubmission.csv file
sample_submission_path = "/content/SampleSubmission.csv"
sample_submission = pd.read_csv(sample_submission_path)

# Display the first few rows of the CSV to understand its structure
sample_submission.head()
