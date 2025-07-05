import pandas as pd
import matplotlib as plt
import seaborn as sns
df = pd.read_csv('dataset/student_wellbeing_final.csv')


# Preview the dataset
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values per column:")
print(missing_values[missing_values > 0]) # i.e display columns with missing only

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

def handle_missing_data(df):
    # Verify required columns exist
    required_columns = ['GPA', 'AttendanceRate (%)', 'StudyHours', 'SleepHours', 'Program']
    assert all(col in df.columns for col in required_columns), \
           f"Missing required columns: {set(required_columns) - set(df.columns)}"
    
    # Store original for comparison
    df_original = df.copy()
    
    # 1. Handle Categorical (Gender)
    df['Gender'] = df['Gender'].fillna('Unknown')
    
    # 2. Prepare for MICE imputation of AttendanceRate
    le = LabelEncoder()
    df['Program_encoded'] = le.fit_transform(df['Program'])
    df['Gender_encoded'] = le.fit_transform(df['Gender'].fillna('Unknown'))  # Handle missing gender
    
    # Select variables to use in MICE model (using GPA instead of Grade)
    mice_vars = ['AttendanceRate (%)', 'GPA', 'StudyHours', 
                'SleepHours', 'Program_encoded']
    
    # 3. Perform MICE imputation
    mice_imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='median'
    )
    
    temp_df = df[mice_vars].copy()
    imputed_values = mice_imputer.fit_transform(temp_df)
    
    # Store imputed values
    df['AttendanceRate_imputed'] = imputed_values[:, 0].clip(0, 100)  # Attendance
    df['GPA_imputed'] = imputed_values[:, 1].clip(0, 4)  # GPA typically 0-4 scale
    
    # 4. Handle other continuous variables with median imputation
    median_cols = ['StudyHours', 'SleepHours', 'ExerciseHours']
    for col in median_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 5. Combine imputed and original values
    df['AttendanceRate (%)'] = df['AttendanceRate (%)'].fillna(df['AttendanceRate_imputed'])
    df['GPA'] = df['GPA'].fillna(df['GPA_imputed'])
    
    # 6. Clean up temporary columns
    df.drop(['Program_encoded', 'Gender_encoded', 
             'AttendanceRate_imputed', 'GPA_imputed'], axis=1, inplace=True)
    
    # Verify no missing values remain
    assert df.isnull().sum().sum() == 0, f"Missing values remain in columns: {df.columns[df.isnull().any()].tolist()}"
    
    return df

# Usage
df_clean = handle_missing_data(df)


def clean_data_types(df):
    """
    Convert data types and ensure consistency across columns.
    Returns a cleaned DataFrame with proper data types.
    """
    # Create copy to avoid SettingWithCopyWarning
    df_clean = df.copy()
    
    # --- 1. Fix Numeric Columns ---
    # Ensure all numeric columns are float (for consistency with potential decimals)
    numeric_cols = ['GPA', 'StudyHours', 'SleepHours', 'ExerciseHours', 'AttendanceRate (%)']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')  # coerce handles non-numeric values
        
    # --- 2. Fix Integer Columns ---
    # SocialMediaHours, MoodLevel, StressLevel should be integers (Likert scale/count data)
    int_cols = ['SocialMediaHours', 'MoodLevel', 'StressLevel']
    for col in int_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')  # Uses pandas' nullable integer type
    
    # --- 3. Handle Categorical Data ---
    # Convert to proper categorical types with consistent formatting
    df_clean['Gender'] = (
        df_clean['Gender']
        .str.strip()  # Remove whitespace
        .str.title()  # Title case (Male/Female/Unknown)
        .astype('category')
    )
    
    df_clean['Program'] = (
        df_clean['Program']
        .str.strip()
        .str.upper()  # Ensure consistent program naming (e.g., "AI" not "ai")
        .astype('category')
    )
    
    # --- 4. Clean Text Columns ---
    df_clean['Name'] = (
        df_clean['Name']
        .str.strip()
        .str.title()  # Standardize name formatting
    )
    
    # --- 5. Validate Age Values ---
    # Ensure ages are realistic for university students
    df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
    df_clean['Age'] = df_clean['Age'].clip(16, 40)  # Reasonable age range for students
    
    # --- 6. Final Consistency Check ---
    # Ensure no columns have mixed types
    type_report = df_clean.dtypes.reset_index()
    type_report.columns = ['Column', 'DataType']
    print("\nFinal Data Types:")
    print(type_report)
    
    return df_clean

# Usage
df_clean = clean_data_types(df_clean)

def add_letter_grades(df):
    """
    Creates a LetterGrade column based on GPA (0-4 scale) with +/- grading scale.
    Uses standard 4.0 GPA scale conversion to letter grades.
    
    GPA to Letter Grade Scale:
    A (3.7-4.0), A– (3.3-3.69), B+ (3.0-3.29), B (2.7-2.99),
    B– (2.3-2.69), C+ (2.0-2.29), C (1.7-1.99), C– (1.3-1.69),
    D+ (1.0-1.29), D (0.7-0.99), D– (0.3-0.69), F (<0.3)
    """
    
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Define GPA to letter grade mapping
    gpa_cutoffs = [
        (3.7, 4.0, 'A'),
        (3.3, 3.69, 'A-'),
        (3.0, 3.29, 'B+'),
        (2.7, 2.99, 'B'),
        (2.3, 2.69, 'B-'),
        (2.0, 2.29, 'C+'),
        (1.7, 1.99, 'C'),
        (1.3, 1.69, 'C-'),
        (1.0, 1.29, 'D+'),
        (0.7, 0.99, 'D'),
        (0.3, 0.69, 'D-'),
        (0.0, 0.29, 'F')
    ]
    
    # Convert to ordered categorical for proper sorting
    grade_order = ['A', 'A-', 'B+', 'B', 'B-', 
                 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F']
    
    # Function to map GPA to letter grade
    def gpa_to_letter(gpa):
        if pd.isna(gpa):  # Handle missing GPAs
            return pd.NA
        for lower, upper, letter in gpa_cutoffs:
            if lower <= gpa <= upper:
                return letter
        return 'F'  # Default for values outside 0-4
    
    # Apply the conversion
    df['LetterGrade'] = df['GPA'].apply(gpa_to_letter)
    
    # Convert to ordered categorical
    df['LetterGrade'] = pd.Categorical(
        df['LetterGrade'],
        categories=grade_order,
        ordered=True
    )
    
    # Validation
    print("\nGrade Distribution:")
    print(df['LetterGrade'].value_counts().sort_index())
    
    # Spot check some values
    test_cases = [4.0, 3.8, 3.5, 3.2, 2.8, 2.5, 2.2, 1.8, 1.5, 1.2, 0.8, 0.5, 0.0]
    print("\nSpot Checks:")
    for gpa in test_cases:
        print(f"GPA: {gpa:.1f} → {gpa_to_letter(gpa)}")
    
    return df

# Usage
df = add_letter_grades(df)

def calculate_wellbeing_score(df):
    """
    Computes a composite WellbeingScore (0-100) from multiple normalized metrics.
    
    Steps:
    1. Normalize each component to 0-1 scale
    2. Apply inverse scaling where needed
    3. Calculate weighted average
    4. Scale final score to 0-100 range
    
    Weights:
    - Sleep: 20%
    - Exercise: 20%
    - Mood: 20%
    - Stress: 20% (inverse)
    - Social Media: 10% (inverse)
    - Study: 10%
    """
    
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    try:
        # --- 1. Normalize each metric ---
        # Sleep (0-9 hours → 0-1)
        df['SleepScore'] = df['SleepHours'].clip(0, 9) / 9
        
        # Exercise (0-7 hours → 0-1)
        df['ExerciseScore'] = df['ExerciseHours'].clip(0, 7) / 7
        
        # Mood (1-10 scale → 0-1)
        df['MoodScore'] = df['MoodLevel'].clip(1, 10) / 10
        
        # Stress (inverse: 1-10 → 0-1)
        df['StressScore'] = 1 - (df['StressLevel'].clip(1, 10) / 10)
        
        # Social Media (inverse: 0-8 hours → 0-1)
        df['SocialMediaScore'] = 1 - (df['SocialMediaHours'].clip(0, 8) / 8)
        
        # Study (0-20 hours → 0-1)
        df['StudyScore'] = df['StudyHours'].clip(0, 20) / 20
        
        # --- 2. Calculate weighted average ---
        weights = {
            'SleepScore': 0.2,
            'ExerciseScore': 0.2,
            'MoodScore': 0.2,
            'StressScore': 0.2,
            'SocialMediaScore': 0.1,
            'StudyScore': 0.1
        }
        
        df['WellbeingScore'] = (
            weights['SleepScore'] * df['SleepScore'] +
            weights['ExerciseScore'] * df['ExerciseScore'] +
            weights['MoodScore'] * df['MoodScore'] +
            weights['StressScore'] * df['StressScore'] +
            weights['SocialMediaScore'] * df['SocialMediaScore'] +
            weights['StudyScore'] * df['StudyScore']
        ) * 100  # Scale to 0-100
        
        # Clip to ensure 0-100 range
        df['WellbeingScore'] = df['WellbeingScore'].clip(0, 100)
        
        # --- 3. Validation ---
        print("\nWellbeing Score Summary:")
        print(df['WellbeingScore'].describe())
        
        print("\nComponent Correlations with Wellbeing:")
        components = ['SleepScore', 'ExerciseScore', 'MoodScore', 
                     'StressScore', 'SocialMediaScore', 'StudyScore']
        print(df[components + ['WellbeingScore']].corr()['WellbeingScore'][:-1])
        
    except KeyError as e:
        print(f"Error: Missing required column - {e}")
        return df
    except Exception as e:
        print(f"Unexpected error: {e}")
        return df
    
    return df

                                      
# First calculate the wellbeing scores
df = calculate_wellbeing_score(df)

# Then create your plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x='WellbeingScore', y='GPA', hue='StressLevel', 
                size='StressLevel', sizes=(20, 200), data=df)
plt.title('GPA vs Wellbeing Score Colored by Stress Level')
plt.show()