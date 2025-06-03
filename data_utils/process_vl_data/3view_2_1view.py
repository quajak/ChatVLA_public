import json

ori_path = '/home/jz08/zhouzy/data/vl_data/0426_text_2025_filter_llava_1w.json'
save_json_path = '/home/jz08/zhouzy/data/vl_data/0426_text_2025_filter_llava_1w_processed.json'
with open(ori_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = []
for item in data:
    new_item = item
    try:
        if isinstance(item['image'], list) and 'real-world' in item['image'][-1]:
            new_item['image']=[item['image'][-1]]
        else:
            new_item['image'] = [item['image']]
    except:
        print("text data")
    new_data.append(new_item)
with open(save_json_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=4)
