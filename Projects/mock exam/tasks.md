## Tasks
**1. Perform the following Data Wrangling tasks in Python in Python (30 marks)**

a) Use pandas to load and preview the dataset (5 marks)
b) Identify and summarize missing values (5 marks)
c) Handle missing values appropriately – use techniques like imputation or dropping with justification (10 marks)
d) Convert data types if necessary and ensure consistency across columns (5 marks)
d) Compute and create new Letter Grade based on the following scale: A (90–100), A– (85–89), B+ (80–84), B (75–79), B– (70–74), C+ (65–69), C (60–64), C– (55–59), D+ (50–54), D (45–49), D– (40–44), F (<40) (5 marks)
e) Computer a new WellbeingScore through normalization and scaling to between 0-100 where the following python code applies

Normalize each metric to a 0–1 scale

    df['SleepScore'] = df['SleepHours'] / 9    # assume 9 is max healthy sleep
    df['ExerciseScore'] = df['ExerciseHours'] / 7 # assume 7 is max healthy sleep
    df['MoodScore'] = df['MoodLevel'] / 10
    df['StressScore'] = 1 - (df['StressLevel'] / 10)  # inverse
    df['SocialMediaScore'] = 1 - (df['SocialMediaHours'] / 8)  # inverse
    df['StudyScore'] = df['StudyHours'] / 20  # normalize for 20-hour study max

Weighted average – you can adjust weights based on importance

    df['WellbeingScore'] = (
        0.2 * df['SleepScore'] +
        0.2 * df['ExerciseScore'] +
        0.2 * df['MoodScore'] +
        0.2 * df['StressScore'] +
        0.1 * df['SocialMediaScore'] +
        0.1 * df['StudyScore']
    ) * 100  # scale to 0–100

**2. Perform Exploratory Data Analysis tasks in Python using seaborn (40 marks)**

a) Conduct univariate analysis. Ensure to use histograms or boxplots (e.g. for Study Hours, Grade, Mood, Stress) (10 marks)
b) Explore bivariate/multivariate relationships using visualizations like scatterplots, heatmaps pair plots and correlation matrix to show relationships between various variables (e.g., Study Hours vs Grade, Sleep vs Exercise vs Mood vs Stress vs Social Media use) (15 marks)
c) Compare academic performance across programs and gender (5 marks)
d) Draw out insights and explain at least 5 meaningful findings you have drawn from the Exploratory Data Analysis and interpretation of patterns (10 marks)

**3.  Develop a dashboard on Power BI or Tableau Public (25 marks)**

a) Load the cleaned dataset and perform basic transformations using Power Query/Tableau prep (5 marks)

b) Include at least the following Visual Charts: (15 marks)
   - KPI Card: Average Grade, Average Attendance Rate
   - Bar chart: Grade by Program and Gender
   - Scatterplot: Grade vs SleepHours or StressLevel
   - Line/Area chart: Relationship between StudyHours and Grade
   - Heatmap/Table: Comparison of Wellbeing Score across Programs
   - Slicer/Filter: Gender or Program

c) Design and create an interactive dashboard and add storytelling through a narrative that brings together an explanation of your dashboard (5 marks)

