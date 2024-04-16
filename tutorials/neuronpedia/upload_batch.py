# Helpers that fix weird NaN stuff
from decimal import Decimal
from typing import Any
import math
import json
import os
import requests
import sys

FEATURE_OUTPUTS_FOLDER = sys.argv[1]


def nanToNeg999(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nanToNeg999(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nanToNeg999(v) for v in obj]
    elif (isinstance(obj, float) or isinstance(obj, Decimal)) and math.isnan(
        obj
    ):
        return -999
    return obj


class NanConverter(json.JSONEncoder):
    def encode(self, o: Any, *args: Any, **kwargs: Any):
        return super().encode(nanToNeg999(o), *args, **kwargs)


# Server info
host = "http://localhost:3000"

# Upload alive features
for file_name in sorted(os.listdir(FEATURE_OUTPUTS_FOLDER)):
    if file_name.startswith("batch-") and file_name.endswith(".json"):
        print("Uploading file: " + file_name)
        file_path = os.path.join(FEATURE_OUTPUTS_FOLDER, file_name)
        f = open(file_path, "r")
        data = json.load(f)

        # Replace NaNs
        data_fixed = json.dumps(data, cls=NanConverter)
        data = json.loads(data_fixed)

        url = host + "/api/local/upload-features"
        resp = requests.post(
            url,
            json=data,
        )
