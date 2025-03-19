import os
import urllib.parse
import requests

# Path to your links.txt file
links_file = '../data/links_clean.txt'

# Function to download files from URLs in links.txt
def download_files(links_file):
    # Read the links from the file
    with open(links_file, 'r') as file:
        links = file.readlines()

    # Process each link
    for link in links:
        link = link.strip()  # Remove any extra spaces or newlines
        if not link:
            continue
        
        # Extract the file name from the URL
        filename = urllib.parse.unquote(link.split("fileName=")[-1])
        filepath = os.path.join('../data', filename)

        print(f"Downloading {filename}...")

        # Send a request to download the file
        response = requests.get(link)
        
        # If the download is successful, save the file
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"{filename} downloaded successfully.")
        else:
            print(f"Failed to download {filename} (Status code: {response.status_code}).")

# Call the function to download files
download_files(links_file)