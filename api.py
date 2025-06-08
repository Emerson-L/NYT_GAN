from pynytimes import NYTAPI
import datetime
import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

#Already pulled: (2022, 1), (2022, 2) 

YEARS_TO_PULL = [2022]
MONTHS_TO_PULL = list(range(1, 13)) #All months
TRAINING_IMAGES_DIR = "training_images"
IMAGE_SIZE = "thumbLarge" #thumbLarge is 150x150, xlarge is bigger, jumbo is bigger

# Set the API key for NYTAPI from some file.txt
def set_api_key(api_key_path):
    with open(api_key_path, "r") as f:
        global nyt
        nyt = NYTAPI(f.read(), parse_dates=True)

# Get image URLs for a specific month and year
def get_image_urls_for_month(year, month):
    data = nyt.archive_metadata(
        date=datetime.datetime(year, month, 1)
    )

    image_urls = {}
    for article in data:
        for item in article["multimedia"]:
            if item["subType"] == "thumbLarge":
                if item["url"]:
                    if article["section_name"] not in image_urls.keys():
                        image_urls[article["section_name"]] = []
                    if item["url"].startswith("http"):
                        image_urls[article["section_name"]].append(item["url"])
                    else:
                        image_urls[article["section_name"]].append("https://www.nytimes.com/" + item["url"])
                    break
    return image_urls


# Download an image from a URL and save it to a specified directory
def download_image(url, directory):    
    response = requests.get(url)
    if response.status_code == 200:
            filename = os.path.join(directory, url.split("/")[-1])
            with open(filename, "wb") as f:
                f.write(response.content)

# For a specified list of months, download images and save them in a structured directory
# Throttled for API rate limits
if __name__ == "__main__":
    set_api_key("api_key.txt")

    if not os.path.exists(TRAINING_IMAGES_DIR):
        os.makedirs(TRAINING_IMAGES_DIR)

    yearmonths = [(year, month) for year in YEARS_TO_PULL for month in MONTHS_TO_PULL]
    if len(yearmonths) == 0:
        raise ValueError("No year/month combinations. Check YEARS_TO_PULL and MONTHS_TO_PULL.")
    if len(yearmonths) > 500:
        raise ValueError("Too many year/month combinations for API rate limit, must be less than 500. Reduce YEARS_TO_PULL or MONTHS_TO_PULL.")

    total_images = sum([len(files) for r, d, files in os.walk(TRAINING_IMAGES_DIR)])
    total_size = sum(file.stat().st_size for file in Path(TRAINING_IMAGES_DIR).rglob('*'))
    print(f"Already downloaded {total_images} images, totaling {total_size / (1024 * 1024):.2f} MB")

    pull_time = None
    for yearmonth in tqdm(yearmonths):
        if pull_time and (datetime.datetime.now() - pull_time).total_seconds() < 12.1:
            time.sleep(12.1 - (datetime.datetime.now() - pull_time).total_seconds())

        urls = (get_image_urls_for_month(yearmonth[0], yearmonth[1]))
        pull_time = datetime.datetime.now()

        for key in tqdm(urls.keys(), leave = False):
            subdirectory = key.replace(" ", "_").lower()
            if not os.path.exists(os.path.join(TRAINING_IMAGES_DIR, subdirectory)):
                os.makedirs(os.path.join(TRAINING_IMAGES_DIR, subdirectory))
            for url in tqdm(urls[key], leave = False):
                download_image(url, directory=os.path.join(TRAINING_IMAGES_DIR, subdirectory))