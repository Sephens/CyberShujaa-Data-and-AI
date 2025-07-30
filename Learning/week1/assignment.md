# Assignment 1: Web Scraping Project

## Overview
**Due Date:** Monday, 19 May 2025, 11:59 PM  
**Objective:** Develop hands-on experience with web scraping using Python in Google Colab.

### Requirements:
- Scrape structured data from [Scrape This Site - Hockey Teams](https://www.scrapethissite.com/pages/forms/)
- Use three key libraries:
  - `requests` for HTTP requests
  - `BeautifulSoup` for HTML parsing
  - `pandas` for data storage/manipulation
- Export results to a CSV file

## Complete Solution with Detailed Explanations

```python
'''
Title: Web Scraping Project
Name: Paula Musuva
Date: 13 May 2025
Description: This script scrapes hockey team statistics from a practice website
and saves the data to a CSV file.
'''

# Import required libraries
from bs4 import BeautifulSoup  # For parsing HTML content
import requests  # For making HTTP requests to websites
import pandas as pd  # For data manipulation and storage

# Set the target URL
url = 'https://www.scrapethissite.com/pages/forms/'
'''
Explanation:
- Stores the website URL we want to scrape in a variable
- This is the page containing hockey team statistics in a table format
'''

# Fetch the webpage content
page = requests.get(url)
'''
Explanation:
- requests.get() sends an HTTP GET request to the specified URL
- Stores the server's response in 'page' variable
- This includes HTML content, status code, headers, etc.
'''

# Parse the HTML content
soup = BeautifulSoup(page.text, 'html.parser')
'''
Explanation:
- BeautifulSoup parses the raw HTML text from the page
- 'html.parser' specifies which parser to use
- Creates a parse tree that we can navigate and search
- 'soup' object now contains the structured HTML document
'''

# Locate the data table
hockey_table = soup.find('table', class_='table')
'''
Explanation:
- Finds the first <table> element with class="table"
- This is where our target data is stored
- The result is a BeautifulSoup Tag object containing the table
'''

# Extract column headers
table_titles = hockey_table.find_all('th')
hockey_table_title = [title.text.strip() for title in table_titles]
'''
Explanation:
1. find_all('th') locates all table header (<th>) elements
2. List comprehension extracts text from each header
3. strip() removes extra whitespace
4. Result is a list of column names for our dataset
'''

# Create empty DataFrame with our column headers
df = pd.DataFrame(columns=hockey_table_title)
'''
Explanation:
- Initializes a pandas DataFrame with our extracted column names
- Creates an empty structure ready to be filled with data
- This ensures our data will be properly organized
'''

# Extract and process table rows
table_data = hockey_table.find_all('tr')
'''
Explanation:
- Finds all table row (<tr>) elements
- The first row contains headers (already processed)
- Remaining rows contain the actual data we want
'''

# Process each data row
for row in table_data[1:]:  # Skip header row
    raw_data = row.find_all('td')
    each_raw_data = [data.text.strip() for data in raw_data]
    '''
    Explanation:
    1. For each row, find all table data (<td>) cells
    2. Extract text from each cell and clean with strip()
    3. Results in a list of values for one team's statistics
    '''
    
    # Add row to DataFrame
    length = len(df)
    df.loc[length] = each_raw_data
    '''
    Explanation:
    1. len(df) gets current DataFrame length (next available index)
    2. df.loc[length] adds a new row at that index
    3. Assigns our extracted data to the new row
    '''

# Display the resulting DataFrame
df
'''
Explanation:
- Shows the complete scraped data in tabular format
- Verifies our scraping was successful
- Columns should match our headers, rows contain all teams
'''

# Export to CSV
df.to_csv(r'./Hockey.csv', index=False)
'''
Explanation:
1. Saves DataFrame to CSV file named 'Hockey.csv'
2. index=False prevents writing row numbers
3. File saves in current working directory
4. Can be opened in Excel or imported to other programs

'''
```

---