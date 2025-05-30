from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep


# URL of what am scrapping
url = 'https://www.scrapethissite.com/pages/forms/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1} to connect...")
        # get the url with a timeout and save it.
        page = requests.get(url, headers=headers, timeout=10)
        page.raise_for_status()
        print("Successfully connected to the website!")
        break
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if attempt < max_retries - 1:
            sleep(retry_delay)
            continue
        raise
# Get the page and access the text and look for specific tags.
soup = BeautifulSoup(page.text, 'html.parser')

# Verify connection by printing page title
print("\nPage Title:", soup.title.text.strip())

# Find the table (based on the tag and the class)
table = soup.find('table', class_='table') # table as a tag and specific class of the table
if table: #if the table has been found
    print("\nFound the hockey stats table!") # print a success
    
    # Extract column headers
    headers = [th.text.strip() for th in table.find_all('th')]
    print("\nColumn Headers:", headers)
    
    # Create DataFrame
    df = pd.DataFrame(columns=headers)
    
    # Extract rows
    for row in table.find_all('tr')[1:5]:  # Just first 5 rows for
        cells = [td.text.strip() for td in row.find_all('td')]
        df.loc[len(df)] = cells
    
    # Show sample data
    print("\nSample Data:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('hockey_stats.csv', index=False)
    print("\nSaved data to hockey_stats.csv!")
else: # if the table has not been found
    print("Error: Could not find the stats table in the page")