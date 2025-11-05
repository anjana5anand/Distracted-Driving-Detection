import requests

base_url = "https://huggingface.co/datasets/flwrlabs/ucf101/resolve/main/data/test-{index}-of-00030.parquet?download=true"

# Loop through 00000 to 00030 (inclusive)
for i in range(31):
    index = f"{i:05d}"
    url = base_url.format(index=index)
    filename = f"test-{index}-of-00030.parquet"
    
    print(f"Downloading {filename}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
    else:
        print(f"Failed to download {filename}, status code: {response.status_code}")
