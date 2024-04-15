import json
import os
import requests
import sys

FEATURE_OUTPUTS_FOLDER = sys.argv[1]

# Server info
host = "http://localhost:3000"

skipped_path = os.path.join(FEATURE_OUTPUTS_FOLDER, "skipped_indexes.json")
f = open(skipped_path, "r")
data = json.load(f)
url = host + "/api/local/upload-dead-features"
resp = requests.post(
    url,
    json=data,
)
