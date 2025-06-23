# Import required libraries
import pandas as pd  # For data manipulation and DataFrames
import numpy as np   # For numerical operations and random number generation
from faker import Faker  # For generating fake personal data
from datetime import datetime, timedelta  # For date handling
import random  # For random number generation

# Initialize Faker with US locale and set seeds for reproducibility
fake = Faker('en_US')  # US-style fake data
fake.seed(42)  # Seed for Faker
np.random.seed(42)  # Seed for NumPy random
random.seed(42)  # Seed for Python random

# Configuration - number of employee records to generate
num_records = 8950

# Define Kenyan counties and their major cities/towns
states_cities = {
    'Nairobi': ['Nairobi', 'Kasarani', 'Ruaraka', 'Embakasi', 'Dagoreti'],
    'Kisumu': ['Kisumu', 'Milimani', 'Kondele', 'Nyalenda', 'Maseno'],
    'Mombasa': ['Mombasa', 'Likoni', 'Kisauni', 'Nyali', 'Changamwe'],
    'Bungoma': ['Bungoma', 'Webuye', 'Kimilili', 'Chwele', 'Naitiri'],
    'Nakuru': ['Nakuru', 'Naivasha', 'Molo', 'Njoro', 'Gilgil'],
    'Kiambu': ['Thika', 'Kiambu', 'Ruiru', 'Kikuyu', 'Limuru'],
    'Uasin Gishu': ['Eldoret', 'Kesses', 'Moiben', 'Soy', 'Turbo'],
    'Kakamega': ['Kakamega', 'Malava', 'Mumias', 'Butere', 'Shinyalu']
}

# Get list of counties
states = list(states_cities.keys())

# Probability distribution for county selection (Nairobi has highest probability)
state_prob = [0.4, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05]

# Assign counties to employees based on probability distribution
assigned_states = np.random.choice(states, size=num_records, p=state_prob)

# For each county assigned, randomly select a city within that county
assigned_cities = [np.random.choice(states_cities[state]) for state in assigned_states]

# Define company departments and their probability distribution
departments = ['HR', 'IT', 'Sales', 'Marketing', 'Finance', 'Operations', 'Customer Service']
departments_prob = [0.02, 0.15, 0.21, 0.08, 0.05, 0.30, 0.19]  # Operations is largest department

# Define job titles for each department
jobtitles = {
    'HR': ['HR Manager', 'HR Coordinator', 'Recruiter', 'HR Assistant'],
    'IT': ['IT Manager', 'Software Developer', 'System Administrator', 'IT Support Specialist'],
    'Sales': ['Sales Manager', 'Sales Consultant', 'Sales Specialist', 'Sales Representative'],
    'Marketing': ['Marketing Manager', 'SEO Specialist', 'Content Creator', 'Marketing Coordinator'],
    'Finance': ['Finance Manager', 'Accountant', 'Financial Analyst', 'Accounts Payable Specialist'],
    'Operations': ['Operations Manager', 'Operations Analyst', 'Logistics Coordinator', 'Inventory Specialist'],
    'Customer Service': ['Customer Service Manager', 'Customer Service Representative', 'Support Specialist', 'Help Desk Technician']
}

# Probability distribution for job titles within each department
jobtitles_prob = {
    'HR': [0.03, 0.3, 0.47, 0.2],  # More assistants than managers
    'IT': [0.02, 0.47, 0.2, 0.31],  # More developers than other roles
    'Sales': [0.03, 0.25, 0.32, 0.4],  # More representatives than managers
    'Marketing': [0.04, 0.25, 0.41, 0.3],  # More content creators than managers
    'Finance': [0.03, 0.37, 0.4, 0.2],  # More analysts than managers
    'Operations': [0.02, 0.2, 0.4, 0.38],  # More coordinators than managers
    'Customer Service': [0.04, 0.3, 0.38, 0.28]  # More representatives than managers
}

# Define education levels
educations = ['High School', "Bachelor", "Master", 'PhD']

# Define required education levels for each job title
education_mapping = {
    'HR Manager': ["Master", "PhD"],
    'HR Coordinator': ["Bachelor", "Master"],
    'Recruiter': ["High School", "Bachelor"],
    'HR Assistant': ["High School", "Bachelor"],
    # ... (similar mappings for other job titles)
}

# Define hiring date distribution by year weights
year_weights = {
    2015: 5,   # 5% of hires
    2016: 8,   # 8%
    2017: 17,   # 17%
    2018: 9,    # 9%
    2019: 10,   # 10%
    2020: 11,   # 11%
    2021: 5,    # 5%
    2022: 12,   # 12%
    2023: 14,   # 14%
    2024: 9     # 9%
}

# Function to generate random dates weighted by year distribution
def generate_custom_date(year_weights):
    # Select year based on weights
    year = random.choices(list(year_weights.keys()), weights=list(year_weights.values()))[0]
    # Generate random date within selected year
    return fake.date_time_between(start_date=datetime(year, 1, 1), end_date=datetime(year, 12, 31))

# Function to generate base salaries by department and job title
def generate_salary(department, job_title):
    # Salary ranges for each position
    salary_dict = {
        'HR': {
            'HR Manager': np.random.randint(60000, 90000),
            'HR Coordinator': np.random.randint(50000, 60000),
            'Recruiter': np.random.randint(50000, 70000),
            'HR Assistant': np.random.randint(50000, 60000)
        },
        # ... (similar salary ranges for other departments)
    }
    return salary_dict[department][job_title]

# Initialize empty list to store employee records
data = []

# Generate each employee record
for _ in range(num_records):
    # Generate employee ID
    employee_id = f"00-{random.randint(10000000, 99999999)}"
    
    # Generate personal details
    first_name = fake.first_name()
    last_name = fake.last_name()
    gender = np.random.choice(['Female', 'Male'], p=[0.46, 0.54])  # 46% female, 54% male
    
    # Generate location details
    state = np.random.choice(states, p=state_prob)
    city = np.random.choice(states_cities[state])
    
    # Generate employment details
    hiredate = generate_custom_date(year_weights)
    department = np.random.choice(departments, p=departments_prob)
    job_title = np.random.choice(jobtitles[department], p=jobtitles_prob[department])
    
    # Generate education and performance details
    education_level = np.random.choice(education_mapping[job_title])
    performance_rating = np.random.choice(
        ['Excellent', 'Good', 'Satisfactory', 'Needs Improvement'], 
        p=[0.12, 0.5, 0.3, 0.08]  # Performance rating distribution
    )
    
    # Generate overtime status
    overtime = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])  # 30% work overtime
    
    # Generate base salary
    salary = generate_salary(department, job_title)
    
    # Add all fields to employee record
    data.append([
        employee_id, first_name, last_name, gender, state, city, hiredate,
        department, job_title, education_level, salary, performance_rating, overtime
    ])

# Define column names for DataFrame
columns = [
    'employee_id', 'first_name', 'last_name', 'gender', 'state', 'city',
    'hiredate', 'department', 'job_title', 'education_level', 'salary',
    'performance_rating', 'overtime'
]

# Create DataFrame from generated data
df = pd.DataFrame(data, columns=columns)

# Function to generate birthdates with realistic age distributions
def generate_birthdate(row):
    # Age group distribution
    age_distribution = {
        'under_25': 0.11,  # 11%
        '25_34': 0.25,     # 25%
        '35_44': 0.31,     # 31%
        '45_54': 0.24,     # 24%
        'over_55': 0.09    # 9%
    }
    
    # Managers and highly educated employees tend to be older
    if any('Manager' in title for title in row['job_title']):
        age = np.random.randint(30, 65)
    elif row['education_level'] == 'PhD':
        age = np.random.randint(27, 65)
    else:
        # Select age group based on distribution
        age_group = np.random.choice(list(age_distribution.keys()), 
                                   p=list(age_distribution.values()))
        # Generate specific age within group
        if age_group == 'under_25':
            age = np.random.randint(20, 25)
        elif age_group == '25_34':
            age = np.random.randint(25, 35)
        elif age_group == '35_44':
            age = np.random.randint(35, 45)
        elif age_group == '45_54':
            age = np.random.randint(45, 55)
        else:
            age = np.random.randint(56, 65)
    
    # Generate birthdate based on calculated age
    birthdate = fake.date_of_birth(minimum_age=age, maximum_age=age)
    return birthdate

# Add birthdate column to DataFrame
df['birthdate'] = df.apply(generate_birthdate, axis=1)

# Configure termination distribution by year
year_weights = {
    2015: 5, 2016: 7, 2017: 10, 2018: 12, 2019: 9,
    2020: 10, 2021: 20, 2022: 10, 2023: 7, 2024: 10
}

# Calculate total terminations (11.2% attrition rate)
total_employees = num_records
termination_percentage = 0.112
total_terminated = int(total_employees * termination_percentage)

# Generate termination dates based on distribution
termination_dates = []
for year, weight in year_weights.items():
    num_terminations = int(total_terminated * (weight / 100))
    termination_dates.extend([year] * num_terminations)

# Shuffle termination years
random.shuffle(termination_dates)

# Assign termination dates to first n employees
terminated_indices = df.index[:total_terminated]
for i, year in enumerate(termination_dates[:total_terminated]):
    df.at[terminated_indices[i], 'termdate'] = datetime(year, 1, 1) + timedelta(days=random.randint(0, 365))

# Set termdate to None for active employees
df['termdate'] = df['termdate'].where(df['termdate'].notnull(), None)

# Ensure termination date is at least 6 months after hire date
df['termdate'] = df.apply(
    lambda row: row['hiredate'] + timedelta(days=180) 
    if row['termdate'] and row['termdate'] < row['hiredate'] + timedelta(days=180) 
    else row['termdate'], 
    axis=1
)

# Define salary multipliers based on education and gender
education_multiplier = {
    'High School': {'Male': 1.03, 'Female': 1.0},  # Males paid 3% more with HS
    "Bachelor": {'Male': 1.115, 'Female': 1.0},    # Males paid 11.5% more with Bachelor's
    "Master": {'Male': 1.0, 'Female': 1.07},       # Females paid 7% more with Master's
    'PhD': {'Male': 1.0, 'Female': 1.17}           # Females paid 17% more with PhD
}

# Function to calculate age from birthdate
def calculate_age(birthdate):
    today = pd.Timestamp('today')
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

# Function to adjust salary based on education, gender, and age
def calculate_adjusted_salary(row):
    base_salary = row['salary']
    gender = row['gender']
    education = row['education_level']
    age = calculate_age(row['birthdate'])

    # Apply education/gender multiplier
    multiplier = education_multiplier.get(education, {}).get(gender, 1.0)
    adjusted_salary = base_salary * multiplier

    # Apply age-based increment (0.1-0.3% per year)
    age_increment = 1 + np.random.uniform(0.001, 0.003) * age
    adjusted_salary *= age_increment

    # Ensure salary doesn't decrease
    adjusted_salary = max(adjusted_salary, base_salary)

    return round(adjusted_salary)  # Round to nearest integer

# Apply salary adjustments
df['salary'] = df.apply(calculate_adjusted_salary, axis=1)

# Convert dates to standard format
df['hiredate'] = pd.to_datetime(df['hiredate']).dt.date
df['birthdate'] = pd.to_datetime(df['birthdate']).dt.date
df['termdate'] = pd.to_datetime(df['termdate']).dt.date

# Display sample data
print(df.head())

# Save to CSV file
df.to_csv('HumanResources.csv', index=False)