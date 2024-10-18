# Nvidia Nemotron 70B Reward Data Cleaner

This script is designed to clean and filter a JSONL dataset based on token count and a reward model score. It processes each line of the dataset, removes lines that exceed a specified token count or fall below a certain reward score threshold, and writes the cleaned data to a new JSONL file. It also provides information about the total tokens and the line with the most tokens in the cleaned file.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python 3.x
- OpenAI Python library
- Tiktoken library
- tqdm library

You can install the required libraries using pip:

```
pip install openai tiktoken tqdm
```

## Configuration

Before running the script, update the following variables in the `__main__` section:

- `input_file`: The path to the input JSONL file containing the dataset to be cleaned.
- `output_file`: The path to the output JSONL file where the cleaned dataset will be written.
- `deleted_file`: The path to the JSONL file where the deleted lines will be written.

Make sure to provide the appropriate file paths based on your setup.

## Adjusting the Deletion Threshold

The script uses a reward model score threshold to determine which lines should be deleted. By default, the threshold is set to -18.5. Lines with a reward score below this threshold will be removed from the dataset.

To change the deletion threshold, locate line 56 in the script:

```python
if reward_score >= -18.5:
```

Modify the value `-18.5` to the desired threshold. For example, if you want to keep lines with a reward score greater than or equal to -20, update the line to:

```python
if reward_score >= -20:
```

Adjusting the threshold allows you to control the level of filtering applied to the dataset based on the reward model score.

## Usage

To run the script, execute the following command:

```
python superclean.py
```

The script will process each line of the input dataset, apply the token count and reward model score filters, and write the cleaned data to the specified output file. It will also write the deleted lines to the specified deleted file.

During the processing, the script will display a progress bar indicating the progress of the dataset cleaning. Once the process is complete, it will print the number of lines kept and removed, as well as the total tokens and the line with the most tokens in the cleaned file.

## Output

The script generates two output files:

1. `output_file`: The cleaned dataset in JSONL format, containing lines that meet the token count and reward score criteria.
2. `deleted_file`: The deleted lines in JSONL format, along with the reason for deletion (token count or reward score) or the actual reward score.

Additionally, the script prints the following information:

- The number of lines kept and removed during the cleaning process.
- The total tokens in the cleaned file.
- The line with the most tokens in the cleaned file.
- The number of tokens in the line with the most tokens.

Use the generated files and the printed information to analyze the cleaned dataset and make further adjustments if necessary.
