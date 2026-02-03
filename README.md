# EMG Data Analysis Script

**User Guide for Tibialis Anterior Dorsiflexion Study**

McGill University - Department of Mechanical Engineering

---

## Overview

This Python script provides comprehensive statistical analysis and visualization for high-density EMG data collected from the tibialis anterior muscle during dorsiflexion tasks. The script automatically processes your data, performs appropriate statistical tests, and generates publication-quality figures.

---

## Features

The script automatically performs the following analyses:

- Loads and cleans your EMG data from CSV files
- Removes outliers and handles missing values
- Calculates descriptive statistics (mean, SD, median, etc.) for each contraction level
- Tests for normality (Shapiro-Wilk test)
- Tests for equal variance across groups (Levene's test)
- Performs appropriate statistical tests (ANOVA or Kruskal-Wallis)
- Conducts pairwise comparisons between contraction levels
- Generates 4 types of publication-quality figures
- Creates comprehensive summary reports

---

## System Requirements

Before using the script, you need:

- Python 3.7 or higher installed on your computer
- Required Python packages (see installation instructions below)
- Your EMG data exported from MUedit as a CSV file
- Basic familiarity with using the command line/terminal

---

## Installation

### Step 1: Install Python Packages

Open your terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

> **Note:** This command installs all the required packages. You only need to do this once.

### Step 2: Download the Analysis Script

Clone this repository or download the `emg_analysis.py` file to a dedicated folder on your computer (e.g., `EMG_Analysis/`).

```bash
git clone [your-repo-url]
cd EMG_Analysis
```

---

## Preparing Your Data

### Data Format Requirements

Your CSV file should be organized with:

- First row containing column headers (variable names)
- One column for participant ID
- One column for grouping (e.g., "contraction_level" with values 5, 10, 20)
- Additional columns for each variable you want to analyze (e.g., firing_rate, motor_unit_count, peak_torque)

### Example Data Structure

| participant_id | contraction_level | firing_rate | motor_unit_count |
|----------------|-------------------|-------------|------------------|
| P01            | 5                 | 8.5         | 12               |
| P01            | 10                | 12.3        | 15               |
| P01            | 20                | 18.7        | 20               |
| P02            | 5                 | 9.1         | 13               |
| ...            | ...               | ...         | ...              |

---

## Usage

### Step 1: Configure the Script

Open the `emg_analysis.py` file in a text editor and locate the `main()` function (around line 800). You need to modify three things:

#### A. Specify your data file:

```python
data_file = 'your_data_file.csv'  # Replace with your actual filename
```

#### B. Specify your grouping column:

```python
group_column = 'contraction_level'  # Replace with your grouping column name
```

#### C. Specify variables to analyze:

```python
variables_to_analyze = [
    'firing_rate',         # Add or remove variables as needed
    'motor_unit_count',
    'peak_torque',
    'average_force'
]
```

### Step 2: Run the Script

1. Place your CSV file in the same folder as `emg_analysis.py`
2. Open terminal/command prompt
3. Navigate to the folder containing the script
4. Run the command:

```bash
python emg_analysis.py
```

5. Wait for the analysis to complete (this may take a few minutes)

---

## Understanding the Output

After running the script, a new folder called `analysis_results/` will be created with the following structure:

```
analysis_results/
├── firing_rate/
│   ├── firing_rate_descriptive_stats.csv
│   ├── firing_rate_pairwise_comparisons.csv
│   ├── firing_rate_boxplot.png
│   ├── firing_rate_barplot.png
│   ├── firing_rate_violinplot.png
│   ├── firing_rate_lineplot.png
│   └── firing_rate_analysis_summary.txt
├── motor_unit_count/
│   └── (same files as above)
└── peak_torque/
    └── (same files as above)
```

### File Descriptions

| File | Description |
|------|-------------|
| `descriptive_stats.csv` | Mean, standard deviation, median, min/max for each contraction level |
| `pairwise_comparisons.csv` | Statistical comparisons between all pairs of contraction levels |
| `boxplot.png` | Shows distribution and outliers for each contraction level |
| `barplot.png` | Shows mean values with error bars (standard error) |
| `violinplot.png` | Shows full probability distribution of data |
| `lineplot.png` | Shows trends across contraction levels |
| `analysis_summary.txt` | Complete text report with all statistical results |

---

## Interpreting the Results

### Statistical Significance

The script uses **p < 0.05** as the threshold for statistical significance. In the pairwise comparisons file, results marked "Yes" in the "Significant (p<0.05)" column indicate a statistically significant difference between those two contraction levels.

### Choosing Between Statistical Tests

The script automatically selects the appropriate statistical test:

- **If data is normally distributed with equal variances:** Uses ANOVA (parametric)
- **If data is NOT normally distributed:** Uses Kruskal-Wallis test (non-parametric)

This decision is based on normality tests and variance tests performed automatically. The summary file will tell you which test was used and why.

---

## Troubleshooting

### Error: "File not found"

**Solution:** Check that your CSV filename exactly matches what you specified in the script, including the file extension (.csv)

### Error: "Column not found"

**Solution:** Verify that the column names in your script match the column headers in your CSV file exactly (case-sensitive)

### No figures appear

**Solution:** Check the `analysis_results` folder - figures are automatically saved as PNG files, not displayed on screen

### Script runs but creates empty folders

**Solution:** Your variable names may not match the columns in your data file. Double-check spelling and capitalization

### Error: "No module named..."

**Solution:** Install the missing package using:

```bash
pip install [package_name]
```

---

## Tips for Best Results

- Always keep a backup of your original data before running any analysis
- Use descriptive column names without spaces (use underscores instead)
- Check the first few rows of your data by opening the CSV in Excel to verify format
- Run the script on a small test dataset first to verify everything works
- Review the `analysis_summary.txt` file for a complete written report of results
- If you modify the script, save it with a new name to preserve the original

---

## Getting Help

If you encounter issues:

1. Check that all software requirements are installed correctly
2. Verify your data file format matches the requirements in the "Preparing Your Data" section
3. Review the error messages - they often indicate exactly what went wrong
4. Consult the inline comments in the script (every line is documented)
5. Contact the lab for technical support
6. Open an issue on this GitHub repository

---

## Citation & Acknowledgments

If you use this script in your research, please acknowledge:

> EMG Analysis Script developed by Chioke Swann, PhD Student, Department of Mechanical Engineering, McGill University, under the supervision of Dr. Guillaume Durandau.

---

## License

[Add your license information here]

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Contact

**Principal Investigator:** Chioke Swann  
**Institution:** McGill University - Department of Mechanical Engineering  
**Supervisor:** Dr. Guillaume Durandau

---

## Version History

- **v1.0** (February 2026) - Initial release
  - Basic statistical analysis
  - Four visualization types
  - Automated test selection
  - Comprehensive reporting

---

## Acknowledgments

- MUedit for motor unit decomposition
- McGill University Department of Mechanical Engineering
- All lab members who contributed to testing and feedback

---

**Document Version:** 1.0  
**Last Updated:** February 2026

---

## Quick Start Example

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# 2. Download the script
git clone [your-repo-url]
cd EMG_Analysis

# 3. Edit the configuration (lines 800-820 in emg_analysis.py)
# Set your CSV filename, grouping column, and variables

# 4. Run the analysis
python emg_analysis.py

# 5. Check your results
cd analysis_results/
```

---
