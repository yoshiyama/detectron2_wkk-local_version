import json

# JSONファイルの読み込み
with open("/home/survey/keikan_bridge/kiritori-henkan3/annotations.json", 'r') as f:
    data = json.load(f)

# すべてのパスについて "\" を "/" に置換
for image in data['JPEGImages']:
    image['file_name'] = image['file_name'].replace("\\", "/")

# 置換後のデータをJSONファイルに書き込み
with open("/home/survey/keikan_bridge/kiritori-henkan3/annotations.json", 'w') as f:
    json.dump(data, f)
