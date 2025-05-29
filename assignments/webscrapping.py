from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep

url = 'https://www.scrapethissite.com/pages/forms/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        print(f"Attempt {attempt + 1} to connect...")
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

soup = BeautifulSoup(page.text, 'html.parser')

# Verify connection by printing page title
print("\nPage Title:", soup.title.text.strip())

# Find the table
table = soup.find('table', class_='table')
if table:
    print("\nFound the hockey stats table!")
    
    # Extract column headers
    headers = [th.text.strip() for th in table.find_all('th')]
    print("\nColumn Headers:", headers)
    
    # Create DataFrame
    df = pd.DataFrame(columns=headers)
    
    # Extract rows
    for row in table.find_all('tr')[1:3]:  # Just first 2 rows for testing
        cells = [td.text.strip() for td in row.find_all('td')]
        df.loc[len(df)] = cells
    
    # Show sample data
    print("\nSample Data:")
    print(df.head())
    
    # Save to CSV
    df.to_csv('hockey_stats.csv', index=False)
    print("\nSaved data to hockey_stats.csv!")
else:
    print("Error: Could not find the stats table in the page")