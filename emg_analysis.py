"""
EMG Data Analysis Script for Tibialis Anterior Study
Author: Chioke Swann, PhD Student
Department of Mechanical Engineering, McGill University

This script analyzes high-density EMG data from the tibialis anterior muscle
during dorsiflexion tasks at different contraction intensities (5%, 10%, 20% MVC).

Input: CSV file containing decomposed motor unit data from MUedit
Output: Statistical analysis results and publication-quality figures
"""

# ============================================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

# Import numpy for numerical operations (arrays, calculations, etc.)
import numpy as np

# Import pandas for data manipulation (loading CSVs, organizing data)
import pandas as pd

# Import matplotlib for creating figures and plots
import matplotlib.pyplot as plt

# Import seaborn for more attractive statistical visualizations
import seaborn as sns

# Import scipy for statistical tests
from scipy import stats
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu, friedman

# Import sklearn for additional statistical methods if needed
from sklearn.preprocessing import StandardScaler

# Import os for file path operations
import os

# Import warnings to suppress unnecessary warning messages
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 2: CONFIGURATION AND SETTINGS
# ============================================================================

# Set the visual style for all plots to make them look professional
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Set default figure size for all plots (width, height in inches)
plt.rcParams['figure.figsize'] = (12, 8)

# Set font to be compatible with scientific publications
plt.rcParams['font.family'] = 'Arial'

# Set DPI (dots per inch) for high-resolution figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Define the contraction intensity levels we're testing
CONTRACTION_LEVELS = [5, 10, 20]  # Percentage of MVC

# Define the number of participants in the study
N_PARTICIPANTS = 12  # 6 male, 6 female as per protocol

# Set significance level for statistical tests (p < 0.05)
ALPHA = 0.05


# ============================================================================
# SECTION 3: DATA LOADING FUNCTION
# ============================================================================

def load_emg_data(filepath):
    """
    Load EMG data from a CSV file exported from MUedit.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing EMG data
        
    Returns:
    --------
    df : pandas DataFrame
        Loaded data with proper column names and data types
    """
    
    # Load the CSV file into a pandas DataFrame
    # This reads the data and organizes it into rows and columns
    df = pd.read_csv(filepath)
    
    # Print the first few rows to verify data loaded correctly
    print("Data loaded successfully!")
    print(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Print column names to verify what data we have
    print("\nColumn names in dataset:")
    print(df.columns.tolist())
    
    # Return the loaded data
    return df


# ============================================================================
# SECTION 4: DATA CLEANING AND PREPARATION
# ============================================================================

def clean_and_prepare_data(df):
    """
    Clean the data by removing outliers and handling missing values.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw data loaded from CSV
        
    Returns:
    --------
    df_clean : pandas DataFrame
        Cleaned and prepared data ready for analysis
    """
    
    # Make a copy of the data so we don't modify the original
    df_clean = df.copy()
    
    # Remove any rows with missing values (NaN)
    # This ensures all our data is complete
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_rows = initial_rows - len(df_clean)
    print(f"\nRemoved {removed_rows} rows with missing values")
    
    # Detect and remove outliers using the IQR (Interquartile Range) method
    # This removes extreme values that might skew our analysis
    print("\nDetecting outliers in numerical columns...")
    
    # Get all numerical columns (not participant ID or categorical variables)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # For each numerical column, identify and remove outliers
    for col in numerical_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        
        # Define outlier boundaries (values outside 1.5 * IQR from Q1 or Q3)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count how many outliers we found
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        # Remove the outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        # Report how many outliers were removed
        if outliers > 0:
            print(f"  {col}: removed {outliers} outliers")
    
    # Print final dataset size
    print(f"\nFinal cleaned dataset: {len(df_clean)} rows")
    
    # Return the cleaned data
    return df_clean


# ============================================================================
# SECTION 5: DESCRIPTIVE STATISTICS
# ============================================================================

def calculate_descriptive_stats(df, group_by_column, value_column):
    """
    Calculate descriptive statistics (mean, SD, median, etc.) for each group.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column name to group by (e.g., 'contraction_level')
    value_column : str
        Column name containing values to analyze (e.g., 'firing_rate')
        
    Returns:
    --------
    stats_df : pandas DataFrame
        Descriptive statistics for each group
    """
    
    # Group the data by the specified column (e.g., contraction level)
    # Then calculate statistics for each group
    stats_df = df.groupby(group_by_column)[value_column].agg([
        ('count', 'count'),          # Number of observations
        ('mean', 'mean'),             # Average value
        ('std', 'std'),               # Standard deviation (spread of data)
        ('sem', 'sem'),               # Standard error of the mean
        ('median', 'median'),         # Middle value
        ('min', 'min'),               # Minimum value
        ('max', 'max'),               # Maximum value
        ('q25', lambda x: x.quantile(0.25)),  # 25th percentile
        ('q75', lambda x: x.quantile(0.75))   # 75th percentile
    ]).round(3)  # Round to 3 decimal places
    
    # Print the results
    print(f"\nDescriptive Statistics for {value_column} by {group_by_column}:")
    print(stats_df)
    
    # Return the statistics table
    return stats_df


# ============================================================================
# SECTION 6: STATISTICAL TESTS FOR NORMALITY
# ============================================================================

def test_normality(df, group_by_column, value_column):
    """
    Test if data follows a normal distribution using Shapiro-Wilk test.
    This helps us decide which statistical tests to use later.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column to group by
    value_column : str
        Column containing values to test
        
    Returns:
    --------
    normality_results : dict
        Dictionary containing test results for each group
    """
    
    print(f"\nTesting normality of {value_column} for each {group_by_column}:")
    print("(If p > 0.05, data is normally distributed)")
    
    # Create a dictionary to store results
    normality_results = {}
    
    # Get unique groups (e.g., 5%, 10%, 20% MVC)
    groups = df[group_by_column].unique()
    
    # Test each group separately
    for group in groups:
        # Get data for this specific group
        group_data = df[df[group_by_column] == group][value_column]
        
        # Perform Shapiro-Wilk test for normality
        # Returns: statistic (W) and p-value
        statistic, p_value = shapiro(group_data)
        
        # Store results
        normality_results[group] = {'statistic': statistic, 'p_value': p_value}
        
        # Print results with interpretation
        if p_value > ALPHA:
            interpretation = "NORMAL distribution"
        else:
            interpretation = "NOT normally distributed"
        
        print(f"  {group}: W = {statistic:.4f}, p = {p_value:.4f} --> {interpretation}")
    
    # Return all results
    return normality_results


# ============================================================================
# SECTION 7: TEST FOR HOMOGENEITY OF VARIANCE
# ============================================================================

def test_homogeneity_of_variance(df, group_by_column, value_column):
    """
    Test if different groups have similar variances using Levene's test.
    This is an assumption for many statistical tests.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column to group by
    value_column : str
        Column containing values to test
        
    Returns:
    --------
    statistic : float
        Test statistic
    p_value : float
        Probability value
    """
    
    # Get unique groups
    groups = df[group_by_column].unique()
    
    # Create a list to hold data from each group
    group_data = []
    
    # Extract data for each group
    for group in groups:
        data = df[df[group_by_column] == group][value_column]
        group_data.append(data)
    
    # Perform Levene's test for homogeneity of variance
    # This tests if all groups have similar spread/variance
    statistic, p_value = levene(*group_data)
    
    # Print results
    print(f"\nLevene's test for homogeneity of variance ({value_column}):")
    print(f"  Statistic = {statistic:.4f}, p = {p_value:.4f}")
    
    if p_value > ALPHA:
        print("  --> Variances are EQUAL across groups (good!)")
    else:
        print("  --> Variances are NOT equal (consider non-parametric tests)")
    
    # Return results
    return statistic, p_value


# ============================================================================
# SECTION 8: COMPARE GROUPS - PARAMETRIC TESTS
# ============================================================================

def anova_test(df, group_by_column, value_column):
    """
    Perform one-way ANOVA to compare means across multiple groups.
    Use this when data is normally distributed with equal variances.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column to group by
    value_column : str
        Column containing values to compare
        
    Returns:
    --------
    f_stat : float
        F-statistic
    p_value : float
        Probability value
    """
    
    # Get unique groups
    groups = df[group_by_column].unique()
    
    # Create a list to hold data from each group
    group_data = []
    
    # Extract data for each group
    for group in groups:
        data = df[df[group_by_column] == group][value_column]
        group_data.append(data)
    
    # Perform one-way ANOVA
    # This tests if at least one group mean is different from the others
    f_stat, p_value = stats.f_oneway(*group_data)
    
    # Print results
    print(f"\nOne-way ANOVA for {value_column} across {group_by_column}:")
    print(f"  F-statistic = {f_stat:.4f}, p = {p_value:.4f}")
    
    if p_value < ALPHA:
        print("  --> Significant difference found between groups!")
    else:
        print("  --> No significant difference between groups")
    
    # Return results
    return f_stat, p_value


# ============================================================================
# SECTION 9: COMPARE GROUPS - NON-PARAMETRIC TESTS
# ============================================================================

def kruskal_wallis_test(df, group_by_column, value_column):
    """
    Perform Kruskal-Wallis test (non-parametric alternative to ANOVA).
    Use this when data is NOT normally distributed.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column to group by
    value_column : str
        Column containing values to compare
        
    Returns:
    --------
    h_stat : float
        H-statistic
    p_value : float
        Probability value
    """
    
    # Get unique groups
    groups = df[group_by_column].unique()
    
    # Create a list to hold data from each group
    group_data = []
    
    # Extract data for each group
    for group in groups:
        data = df[df[group_by_column] == group][value_column]
        group_data.append(data)
    
    # Perform Kruskal-Wallis test
    # This is the non-parametric version of ANOVA
    h_stat, p_value = kruskal(*group_data)
    
    # Print results
    print(f"\nKruskal-Wallis test for {value_column} across {group_by_column}:")
    print(f"  H-statistic = {h_stat:.4f}, p = {p_value:.4f}")
    
    if p_value < ALPHA:
        print("  --> Significant difference found between groups!")
    else:
        print("  --> No significant difference between groups")
    
    # Return results
    return h_stat, p_value


# ============================================================================
# SECTION 10: POST-HOC PAIRWISE COMPARISONS
# ============================================================================

def pairwise_comparisons(df, group_by_column, value_column, parametric=True):
    """
    Perform pairwise comparisons between all groups.
    This tells us which specific groups are different from each other.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data
    group_by_column : str
        Column to group by
    value_column : str
        Column containing values to compare
    parametric : bool
        If True, use t-test; if False, use Mann-Whitney U test
        
    Returns:
    --------
    comparison_results : pandas DataFrame
        Results of all pairwise comparisons
    """
    
    # Get unique groups
    groups = df[group_by_column].unique()
    
    # Create lists to store comparison results
    group1_list = []
    group2_list = []
    statistic_list = []
    p_value_list = []
    significant_list = []
    
    # Compare each pair of groups
    # Using nested loops to get all unique pairs
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:  # Only compare each pair once
            # Get data for both groups
            data1 = df[df[group_by_column] == group1][value_column]
            data2 = df[df[group_by_column] == group2][value_column]
            
            if parametric:
                # Use independent t-test (parametric)
                statistic, p_value = stats.ttest_ind(data1, data2)
                test_name = "t-test"
            else:
                # Use Mann-Whitney U test (non-parametric)
                statistic, p_value = mannwhitneyu(data1, data2)
                test_name = "Mann-Whitney U"
            
            # Store results
            group1_list.append(group1)
            group2_list.append(group2)
            statistic_list.append(statistic)
            p_value_list.append(p_value)
            
            # Check if significant
            if p_value < ALPHA:
                significant_list.append("Yes")
            else:
                significant_list.append("No")
    
    # Create a DataFrame with all comparison results
    comparison_results = pd.DataFrame({
        'Group 1': group1_list,
        'Group 2': group2_list,
        'Statistic': statistic_list,
        'p-value': p_value_list,
        'Significant (p<0.05)': significant_list
    })
    
    # Round numeric values for readability
    comparison_results['Statistic'] = comparison_results['Statistic'].round(4)
    comparison_results['p-value'] = comparison_results['p-value'].round(4)
    
    # Print results
    print(f"\nPairwise comparisons ({test_name}) for {value_column}:")
    print(comparison_results)
    
    # Return the results table
    return comparison_results


# ============================================================================
# SECTION 11: VISUALIZATION - BOX PLOTS
# ============================================================================

def create_boxplot(df, group_by_column, value_column, title, ylabel, save_path=None):
    """
    Create a box plot to visualize the distribution of data across groups.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    group_by_column : str
        Column to group by (x-axis)
    value_column : str
        Column containing values to plot (y-axis)
    title : str
        Title for the plot
    ylabel : str
        Label for y-axis
    save_path : str, optional
        Path to save the figure (if None, just displays)
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create box plot using seaborn
    # Box plots show median, quartiles, and outliers
    sns.boxplot(data=df, x=group_by_column, y=value_column, palette='Set2')
    
    # Add individual data points on top of boxes
    # This shows the actual distribution of data
    sns.stripplot(data=df, x=group_by_column, y=value_column, 
                  color='black', alpha=0.3, size=4)
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(group_by_column.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add a grid for easier reading
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save or display the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBox plot saved to: {save_path}")
    else:
        plt.show()
    
    # Close the figure to free memory
    plt.close()


# ============================================================================
# SECTION 12: VISUALIZATION - BAR PLOTS WITH ERROR BARS
# ============================================================================

def create_barplot_with_error(df, group_by_column, value_column, title, ylabel, save_path=None):
    """
    Create a bar plot showing mean values with error bars (standard error).
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    group_by_column : str
        Column to group by (x-axis)
    value_column : str
        Column containing values to plot (y-axis)
    title : str
        Title for the plot
    ylabel : str
        Label for y-axis
    save_path : str, optional
        Path to save the figure
    """
    
    # Calculate mean and standard error for each group
    summary_stats = df.groupby(group_by_column)[value_column].agg(['mean', 'sem']).reset_index()
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    # capsize adds caps to the error bars
    plt.bar(summary_stats[group_by_column], summary_stats['mean'], 
            yerr=summary_stats['sem'], capsize=5, alpha=0.7, 
            color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(group_by_column.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add a grid for easier reading
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBar plot saved to: {save_path}")
    else:
        plt.show()
    
    # Close figure
    plt.close()


# ============================================================================
# SECTION 13: VISUALIZATION - VIOLIN PLOTS
# ============================================================================

def create_violinplot(df, group_by_column, value_column, title, ylabel, save_path=None):
    """
    Create a violin plot to show the full distribution of data.
    Violin plots combine box plots with kernel density estimation.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    group_by_column : str
        Column to group by (x-axis)
    value_column : str
        Column containing values to plot (y-axis)
    title : str
        Title for the plot
    ylabel : str
        Label for y-axis
    save_path : str, optional
        Path to save the figure
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create violin plot
    # Violin plots show the probability density of data at different values
    sns.violinplot(data=df, x=group_by_column, y=value_column, palette='muted')
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(group_by_column.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nViolin plot saved to: {save_path}")
    else:
        plt.show()
    
    # Close figure
    plt.close()


# ============================================================================
# SECTION 14: VISUALIZATION - LINE PLOT FOR TRENDS
# ============================================================================

def create_lineplot(df, group_by_column, value_column, title, ylabel, save_path=None):
    """
    Create a line plot showing mean values across groups with confidence intervals.
    Useful for showing trends across contraction levels.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to plot
    group_by_column : str
        Column to group by (x-axis)
    value_column : str
        Column containing values to plot (y-axis)
    title : str
        Title for the plot
    ylabel : str
        Label for y-axis
    save_path : str, optional
        Path to save the figure
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create line plot with confidence interval shading
    # ci=95 shows 95% confidence interval
    sns.lineplot(data=df, x=group_by_column, y=value_column, 
                 marker='o', markersize=10, linewidth=2.5, ci=95)
    
    # Add title and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(group_by_column.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nLine plot saved to: {save_path}")
    else:
        plt.show()
    
    # Close figure
    plt.close()


# ============================================================================
# SECTION 15: COMPREHENSIVE ANALYSIS REPORT
# ============================================================================

def generate_analysis_report(df, group_by_column, value_column, output_dir='results'):
    """
    Run a complete statistical analysis and generate all visualizations.
    This is the main function that ties everything together.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned data ready for analysis
    group_by_column : str
        Column to group by (e.g., 'contraction_level')
    value_column : str
        Column to analyze (e.g., 'firing_rate', 'motor_unit_count', etc.)
    output_dir : str
        Directory to save results and figures
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE ANALYSIS REPORT FOR: {value_column}")
    print("="*80)
    
    # Step 1: Calculate descriptive statistics
    print("\n" + "-"*80)
    print("STEP 1: DESCRIPTIVE STATISTICS")
    print("-"*80)
    descriptive_stats = calculate_descriptive_stats(df, group_by_column, value_column)
    
    # Save descriptive statistics to CSV
    stats_file = os.path.join(output_dir, f'{value_column}_descriptive_stats.csv')
    descriptive_stats.to_csv(stats_file)
    print(f"\nDescriptive statistics saved to: {stats_file}")
    
    # Step 2: Test for normality
    print("\n" + "-"*80)
    print("STEP 2: TEST FOR NORMALITY")
    print("-"*80)
    normality_results = test_normality(df, group_by_column, value_column)
    
    # Determine if data is normally distributed across all groups
    all_normal = all(result['p_value'] > ALPHA for result in normality_results.values())
    
    # Step 3: Test for homogeneity of variance
    print("\n" + "-"*80)
    print("STEP 3: TEST FOR HOMOGENEITY OF VARIANCE")
    print("-"*80)
    levene_stat, levene_p = test_homogeneity_of_variance(df, group_by_column, value_column)
    
    # Determine if variances are equal
    equal_variance = levene_p > ALPHA
    
    # Step 4: Choose and perform appropriate statistical test
    print("\n" + "-"*80)
    print("STEP 4: STATISTICAL COMPARISON BETWEEN GROUPS")
    print("-"*80)
    
    if all_normal and equal_variance:
        # Use parametric test (ANOVA)
        print("\nData meets assumptions for parametric testing")
        print("Using ONE-WAY ANOVA...")
        f_stat, anova_p = anova_test(df, group_by_column, value_column)
        use_parametric = True
    else:
        # Use non-parametric test (Kruskal-Wallis)
        print("\nData does NOT meet assumptions for parametric testing")
        print("Using KRUSKAL-WALLIS test...")
        h_stat, kw_p = kruskal_wallis_test(df, group_by_column, value_column)
        use_parametric = False
    
    # Step 5: Post-hoc pairwise comparisons (if main test is significant)
    print("\n" + "-"*80)
    print("STEP 5: POST-HOC PAIRWISE COMPARISONS")
    print("-"*80)
    
    # Check if we should do pairwise comparisons
    main_test_p = anova_p if use_parametric else kw_p
    
    if main_test_p < ALPHA:
        print("\nMain test was significant, performing pairwise comparisons...")
        pairwise_results = pairwise_comparisons(df, group_by_column, value_column, 
                                                parametric=use_parametric)
        
        # Save pairwise comparison results
        pairwise_file = os.path.join(output_dir, f'{value_column}_pairwise_comparisons.csv')
        pairwise_results.to_csv(pairwise_file, index=False)
        print(f"\nPairwise comparison results saved to: {pairwise_file}")
    else:
        print("\nMain test was NOT significant, skipping pairwise comparisons")
    
    # Step 6: Generate all visualizations
    print("\n" + "-"*80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # Create box plot
    boxplot_path = os.path.join(output_dir, f'{value_column}_boxplot.png')
    create_boxplot(df, group_by_column, value_column, 
                   f'Distribution of {value_column} by {group_by_column}',
                   value_column.replace('_', ' ').title(),
                   save_path=boxplot_path)
    
    # Create bar plot with error bars
    barplot_path = os.path.join(output_dir, f'{value_column}_barplot.png')
    create_barplot_with_error(df, group_by_column, value_column,
                              f'Mean {value_column} by {group_by_column}',
                              value_column.replace('_', ' ').title(),
                              save_path=barplot_path)
    
    # Create violin plot
    violinplot_path = os.path.join(output_dir, f'{value_column}_violinplot.png')
    create_violinplot(df, group_by_column, value_column,
                      f'Distribution of {value_column} by {group_by_column}',
                      value_column.replace('_', ' ').title(),
                      save_path=violinplot_path)
    
    # Create line plot
    lineplot_path = os.path.join(output_dir, f'{value_column}_lineplot.png')
    create_lineplot(df, group_by_column, value_column,
                    f'Trend of {value_column} across {group_by_column}',
                    value_column.replace('_', ' ').title(),
                    save_path=lineplot_path)
    
    # Step 7: Create summary report
    print("\n" + "-"*80)
    print("STEP 7: GENERATING SUMMARY REPORT")
    print("-"*80)
    
    # Create a text file with summary of all results
    summary_file = os.path.join(output_dir, f'{value_column}_analysis_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"STATISTICAL ANALYSIS SUMMARY: {value_column}\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. DESCRIPTIVE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(descriptive_stats.to_string())
        f.write("\n\n")
        
        f.write("2. NORMALITY TEST (Shapiro-Wilk)\n")
        f.write("-"*80 + "\n")
        for group, result in normality_results.items():
            f.write(f"  {group}: W = {result['statistic']:.4f}, p = {result['p_value']:.4f}\n")
        f.write(f"\nAll groups normally distributed: {all_normal}\n\n")
        
        f.write("3. HOMOGENEITY OF VARIANCE TEST (Levene)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Statistic = {levene_stat:.4f}, p = {levene_p:.4f}\n")
        f.write(f"  Equal variances: {equal_variance}\n\n")
        
        f.write("4. MAIN STATISTICAL TEST\n")
        f.write("-"*80 + "\n")
        if use_parametric:
            f.write(f"  Test used: One-way ANOVA\n")
            f.write(f"  F-statistic = {f_stat:.4f}, p = {anova_p:.4f}\n")
        else:
            f.write(f"  Test used: Kruskal-Wallis\n")
            f.write(f"  H-statistic = {h_stat:.4f}, p = {kw_p:.4f}\n")
        
        f.write(f"  Significant difference: {main_test_p < ALPHA}\n\n")
        
        if main_test_p < ALPHA:
            f.write("5. POST-HOC PAIRWISE COMPARISONS\n")
            f.write("-"*80 + "\n")
            f.write(pairwise_results.to_string(index=False))
            f.write("\n")
    
    print(f"\nSummary report saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")


# ============================================================================
# SECTION 16: MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to run the entire analysis pipeline.
    Modify the parameters below to match your data.
    """
    
    print("\n" + "="*80)
    print("EMG DATA ANALYSIS PIPELINE")
    print("Tibialis Anterior - Dorsiflexion Study")
    print("="*80)
    
    # ========================================================================
    # STEP 1: SPECIFY YOUR DATA FILE PATH
    # ========================================================================
    
    # CHANGE THIS to the path of your CSV file
    data_file = 'emg_data.csv'  # <<< PUT YOUR FILE NAME HERE
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"\nERROR: File '{data_file}' not found!")
        print("Please update the 'data_file' variable with the correct path.")
        return
    
    # ========================================================================
    # STEP 2: LOAD THE DATA
    # ========================================================================
    
    print("\n" + "-"*80)
    print("LOADING DATA")
    print("-"*80)
    
    # Load data from CSV file
    df = load_emg_data(data_file)
    
    # ========================================================================
    # STEP 3: CLEAN THE DATA
    # ========================================================================
    
    print("\n" + "-"*80)
    print("CLEANING DATA")
    print("-"*80)
    
    # Clean the data (remove outliers and missing values)
    df_clean = clean_and_prepare_data(df)
    
    # ========================================================================
    # STEP 4: SPECIFY WHICH VARIABLES TO ANALYZE
    # ========================================================================
    
    # CHANGE THESE to match your actual column names
    # The column that indicates which group (e.g., 5%, 10%, 20% MVC)
    group_column = 'contraction_level'  # <<< PUT YOUR GROUPING COLUMN HERE
    
    # The columns you want to analyze (e.g., firing rate, motor unit count)
    # You can add or remove variables from this list
    variables_to_analyze = [
        'firing_rate',         # <<< PUT YOUR VARIABLE NAMES HERE
        'motor_unit_count',
        'peak_torque',
        'average_force'
    ]
    
    # ========================================================================
    # STEP 5: RUN ANALYSIS FOR EACH VARIABLE
    # ========================================================================
    
    # Create a main results directory
    main_output_dir = 'analysis_results'
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
    
    # Analyze each variable
    for variable in variables_to_analyze:
        # Check if this column exists in the data
        if variable not in df_clean.columns:
            print(f"\nWARNING: Column '{variable}' not found in data. Skipping...")
            continue
        
        # Create a subdirectory for this variable
        variable_output_dir = os.path.join(main_output_dir, variable)
        
        # Run complete analysis for this variable
        generate_analysis_report(df_clean, group_column, variable, 
                                output_dir=variable_output_dir)
    
    # ========================================================================
    # STEP 6: ANALYSIS COMPLETE
    # ========================================================================
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {main_output_dir}/")
    print("\nEach variable has its own folder containing:")
    print("  - Descriptive statistics (CSV)")
    print("  - Statistical test results (CSV)")
    print("  - Multiple visualization plots (PNG)")
    print("  - Comprehensive summary report (TXT)")


# ============================================================================
# SECTION 17: RUN THE SCRIPT
# ============================================================================

# This section runs only when you execute this script directly
# It won't run if you import this script as a module
if __name__ == "__main__":
    # Run the main analysis function
    main()


# ============================================================================
# END OF SCRIPT
# ============================================================================

"""
INSTRUCTIONS FOR USE:
=====================

1. Save this script as 'emg_analysis.py' in your project folder

2. Prepare your CSV data file with these requirements:
   - First row should contain column headers
   - Include a column for grouping (e.g., 'contraction_level' with values 5, 10, 20)
   - Include columns for the variables you want to analyze
   - Example columns: participant_id, contraction_level, firing_rate, motor_unit_count, etc.

3. Edit the main() function (around line 800) to specify:
   - Your CSV file path (data_file variable)
   - Your grouping column name (group_column variable)
   - Your variable names to analyze (variables_to_analyze list)

4. Install required packages (if not already installed):
   Open command prompt/terminal and run:
   pip install numpy pandas matplotlib seaborn scipy scikit-learn

5. Run the script:
   Open command prompt/terminal in the folder containing this script
   Type: python emg_analysis.py
   Press Enter

6. Check the results:
   - A new folder called 'analysis_results' will be created
   - Inside will be subfolders for each variable you analyzed
   - Each subfolder contains statistics, plots, and a summary report

TROUBLESHOOTING:
===============

- If you get "File not found" error: Check that your CSV file name is correct
- If you get "Column not found" error: Check that your column names match exactly
- If plots don't save: Make sure you have write permissions in the folder
- If you get import errors: Install the required packages (see step 4 above)

For questions, refer to the inline comments throughout the script.
Each function is thoroughly documented with what it does and how to use it.
"""
