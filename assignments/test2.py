import kagglehub

# Download latest version
path = kagglehub.dataset_download("asimmahmudov/top-rated-coffee")

print("Path to dataset files:", path)