# Medical Report Summarization with BART

This project implements a Python script that extracts and summarizes medical reports based on a given patient ID using **BART** (Bidirectional and Auto-Regressive Transformers). The script retrieves medical information from a dataset and generates concise summaries for key fields like **Diagnosis**, **Medical History**, **Prescribed Medication**, and **Lab Test Results**.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To get started, you need to clone the repository and install the required dependencies.

1. Clone the repository:

```bash
git clone https://github.com/yourusername/medical-report-summarization.git
cd medical-report-summarization
```

2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset contains medical information for 5000 patients with the following columns:

- **Patient ID**: Unique integer ID of the patient.
- **Name**: Name of the patient.
- **Age**: Age of the patient.
- **Gender**: Gender of the patient.
- **Diagnosis**: Medical diagnosis text.
- **Medical History**: Description of the patient's medical history.
- **Prescribed Medication**: List of prescribed medications (some entries may be missing).
- **Lab Test Results**: Description of the patient's lab test results.

> **Note**: Ensure you have the dataset in a CSV format and place it in the same directory as the script.

## Usage

1. Run the script:

```bash
python medical_report_summarizer.py
```

2. The script will ask you to input a **Patient ID** (between 1 and 5000). If the ID is invalid or out of range, it will prompt you to enter a valid ID.
3. After retrieving the data for the given ID, it will show both the full patient information and a summarized version of the medical report.

> You can continue searching for more Patient IDs or end the process by typing `exit` when prompted.

## How it Works

The script performs the following tasks:

1. **Input**: The user provides a Patient ID.
2. **Search**: The script searches the dataset for the corresponding Patient ID.
3. **Data Extraction**: It extracts relevant medical information.
4. **Summarization**: Using the **BART** model, it generates summaries for key fields like Diagnosis, Medical History, Medication, and Lab Test Results.
5. **Output**: Displays the full information of the patient and a summarized version of their medical reports.

## Dependencies

- **transformers**: For BART model and summarization pipeline.
- **pandas**: For dataset manipulation.
- **torch**: For model inference.
- **matplotlib**: For displaying results in a tabular format.

To install the dependencies, run:

```bash
pip install transformers pandas torch matplotlib
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
