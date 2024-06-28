# Description: This script scrapes the content of a Loeb Classical Library text and saves it to a text file.


import requests
from bs4 import BeautifulSoup


def scrape_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve page {url} with status code {response.status_code}")
        return None

def parse_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find('div', id='contentRoot')
    if content:
        return content.text.strip()
    return None

def save_content(page_number, content):
    filename = f"page_{page_number}.txt"
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Saved content of page {page_number} to {filename}")

def main():
    base_url1 = "https://www.loebclassics.com/view/arrian-anabasis_alexander/1976/pb_LCL236."
    base_url2 = "https://www.loebclassics.com/view/LCL236/1976/pb_LCL236."
    page_number = 451
    
    while True:
        if page_number < 451:
            url = f"{base_url1}{page_number}.xml"
        else:
            url = f"{base_url2}{page_number}.xml"
        
        html_content = scrape_page(url)
        if not html_content:
            break
        
        content = parse_content(html_content)
        if content:
            save_content(page_number, content)
        else:
            print(f"No content found on page {page_number}")

        page_number += 2

if __name__ == "__main__":
    main()

import requests
from bs4 import BeautifulSoup
import os
import time

def scrape_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    elif response.status_code == 429:  # Too Many Requests
        print(f"Too many requests. Retrying after 60 seconds...")
        time.sleep(60)
        return scrape_page(url)
    else:
        print(f"Failed to retrieve page {url} with status code {response.status_code}")
        return None

def parse_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find('div', id='contentRoot')
    if content:
        return content.text.strip()
    return None

def save_content(volume, book, page_number, content):
    filename = f"DIODORUS_SICULUS_Volume_{volume}_Book_{book}_Page_{page_number}.txt"
    with open(filename, 'w') as file:
        file.write(content)
    print(f"Saved content of volume {volume}, book {book}, page {page_number} to {filename}")

def file_exists(volume, book, page_number):
    filename = f"DIODORUS_SICULUS_Volume_{volume}_Book_{book}_Page_{page_number}.txt"
    return os.path.isfile(filename)

def scrape_and_save(volume, book, start_page, end_page, base_url):
    page_number = start_page
    while page_number <= end_page:
        skip = False
        if file_exists(volume, book, page_number):
            print(f"File for volume {volume}, book {book}, page {page_number} already exists. Skipping.")
            skip = True
        else:
            url = f"{base_url}{page_number}.xml"
            html_content = scrape_page(url)
            
            if html_content:
                content = parse_content(html_content)
                if content:
                    save_content(volume, book, page_number, content)
                else:
                    print(f"No content found on volume {volume}, book {book}, page {page_number}")
            else:
                print(f"Page {page_number} not found. Skipping to the next page.")

        page_number += 2
        if not skip:
            time.sleep(2)  # Delay between requests to avoid rate limits

def main():
    sources = [
        {"volume": "VII", "book": "XV", "start_page": 1, "end_page": 222, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL389."},
        {"volume": "VII", "book": "XVI", "start_page": 223, "end_page": 425, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL389."},
        {"volume": "VIII", "book": "XVI.66-95", "start_page": 21, "end_page": 104, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL422."},
        {"volume": "VIII", "book": "XVII", "start_page": 105, "end_page": 472, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL422."},
        {"volume": "VIII", "book": "ADDENDA", "start_page": 473, "end_page": 475, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL422."},
        {"volume": "IX", "book": "XVIII", "start_page": 3, "end_page": 217, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL377."},
        {"volume": "IX", "book": "XIX.I-65", "start_page": 219, "end_page": 413, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL377."},
        {"volume": "IX", "book": "INDEX_OF_NAMES", "start_page": 414, "end_page": 415, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL377."},
        {"volume": "X", "book": "XIX.66-110", "start_page": 1, "end_page": 134, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."},
        {"volume": "X", "book": "XX", "start_page": 135, "end_page": 445, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."},
        {"volume": "X", "book": "INDEX_OF_NAMES", "start_page": 446, "end_page": 453, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."},
        {"volume": "X", "book": "MAPS_ITALY_SICILY_NORTH_AFRICA", "start_page": 454, "end_page": 455, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."},
        {"volume": "X", "book": "MAPS_HELLENISTIC_MONARCHIES", "start_page": 456, "end_page": 457, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."},
        {"volume": "X", "book": "MAPS_GREECE_AEGEAN_ASIA_MINOR", "start_page": 458, "end_page": 999, "base_url": "https://www.loebclassics.com/view/diodorus_siculus-library_history/1933/pb_LCL390."}
    ]
    
    for source in sources:
        scrape_and_save(source["volume"], source["book"], source["start_page"], source["end_page"], source["base_url"])

if __name__ == "__main__":
    main()

# import requests
# from bs4 import BeautifulSoup
# import os
# import time

# def scrape_page(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.text
#     elif response.status_code == 429:  # Too Many Requests
#         print(f"Too many requests. Retrying after 60 seconds...")
#         time.sleep(60)
#         return scrape_page(url)
#     else:
#         print(f"Failed to retrieve page {url} with status code {response.status_code}")
#         return None

# def parse_content(html_content):
#     soup = BeautifulSoup(html_content, 'html.parser')
#     content = soup.find('div', id='contentRoot')
#     if content:
#         return content.text.strip()
#     return None

# def save_content(page_number, content):
#     directory = "plutarch_lives_alexander"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     filename = os.path.join(directory, f"Page_{page_number}.txt")
#     with open(filename, 'w') as file:
#         file.write(content)
#     print(f"Saved content of page {page_number} to {filename}")

# def file_exists(page_number):
#     directory = "plutarch_lives_alexander"
#     filename = os.path.join(directory, f"Page_{page_number}.txt")
#     return os.path.isfile(filename)

# def scrape_and_save(start_page, end_page, base_url):
#     page_number = start_page
#     while page_number <= end_page:
#         if file_exists(page_number):
#             print(f"File for page {page_number} already exists. Skipping.")
#         else:
#             url = f"{base_url}{page_number}.xml"
#             html_content = scrape_page(url)
            
#             if html_content:
#                 content = parse_content(html_content)
#                 if content:
#                     save_content(page_number, content)
#                 else:
#                     print(f"No content found on page {page_number}")
#             else:
#                 print(f"Page {page_number} not found. Skipping to the next page.")

#         page_number += 2
#         time.sleep(2)  # Delay between requests to avoid rate limits

# def main():
#     start_page = 227
#     end_page = 435
#     base_url = "https://www.loebclassics.com/view/plutarch-lives_alexander/1919/pb_LCL099."
    
#     scrape_and_save(start_page, end_page, base_url)

# if __name__ == "__main__":
#     main()
