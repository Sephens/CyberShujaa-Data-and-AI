from bs4 import BeautifulSoup
import requests
import pandas as pd

# URL of what am scrapping
url = 'https://www.scrapethissite.com/pages/forms/'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
table = soup.find('table', class_='table')

headers = [th.text.strip() for th in table.find_all('th')]
print("\nColumn Headers:", headers)

# create a DataFrame
df = pd.DataFrame(columns=headers)

# extract rows

for row in table.find_all('tr')[1:5]:  # Just first 5 rows for
    cells = [td.text.strip() for td in row.find_all('td')]
    df.loc[len(df)] = cells

# show sample table
print(df.head())

#save to excel
df.to_excel('text.xlsx', index=False)

df.to_json('test.json')