import json
from pprint import pprint
json_file=open('data.json')

data = json.load(json_file)
pprint(data)
json_file.close
