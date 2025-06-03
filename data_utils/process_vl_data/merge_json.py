import json
import random
random.seed(0)

data_list = []
json_list = [
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_math_0313.json',#198
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_text_only_plus_0318.json',#1600
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_text_only_plus_0412.json',#2025 OK
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_single_ocr_0313.json',#18
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_llava_ocr_has_number_0316.json',#17830
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/processed_ocr_three_below_0317.json',#20
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/verification_code_0402.json',#2025
    # '/home/jz08/zhouzy/code/moevla/data_utils/process_vl_data/verification_code_0410.json',#2025
    '/home/jz08/zhouzy/data/vl_data/llava_v1_5_mix665k.json'
]
# merge_ratio = [10,1.2,10,1]
# merge_ratio = [1,0.118,1]
# target = [1,2025,1]
merge_ratio = [0]
target = [1000000]
use_target = True
save_json_path = '/home/jz08/zhouzy/data/vl_data/0503_cutted_llava.json'

# Maximum number of conversation turns (pairs of human-assistant messages)
MAX_CONVERSATION_TURNS = 5
# Maximum length of any single message in the conversation
MAX_MESSAGE_LENGTH = 500

def random_select_conversations(data_item):
    """Select conversations with the first pair always included and randomly select the rest."""
    conversations = data_item.get('conversations', [])
    
    # If we have more than MAX_CONVERSATION_TURNS*2 messages, select first pair + random others
    if len(conversations) > MAX_CONVERSATION_TURNS * 2:
        # Convert to pairs (assuming they come in human-assistant pairs)
        pairs = [(conversations[i], conversations[i+1]) 
                 for i in range(0, len(conversations), 2) 
                 if i+1 < len(conversations)]
        
        # Always include the first pair
        first_pair = pairs[0]
        remaining_pairs = pairs[1:]
        
        # Randomly select (MAX_CONVERSATION_TURNS-1) pairs from the remaining
        if remaining_pairs and MAX_CONVERSATION_TURNS > 1:
            selected_pairs = random.sample(remaining_pairs, 
                                          min(MAX_CONVERSATION_TURNS-1, len(remaining_pairs)))
            selected_pairs.insert(0, first_pair)  # Add first pair at the beginning
        else:
            selected_pairs = [first_pair]  # Only first pair if no others or MAX_CONVERSATION_TURNS is 1
        
        # Flatten the pairs back into a list
        new_conversations = []
        for pair in selected_pairs:
            new_conversations.extend(pair)
        
        data_item['conversations'] = new_conversations
        del remaining_pairs
        del first_pair
        del selected_pairs
    
    # Check and truncate message length if needed
    for conv in data_item['conversations']:
        if len(conv.get('value', '')) > MAX_MESSAGE_LENGTH:
            conv['value'] = conv['value'][:MAX_MESSAGE_LENGTH]
            
    return data_item

for i,json_file in enumerate(json_list):
    with open(json_file, 'r', encoding='utf-8') as f:
        data=json.load(f)
    print(f"Original length of {json_file}: {len(data)}")
    
    # Apply random conversation selection to each item
    data = [random_select_conversations(item) for item in data]
    print(f"Processed conversations by randomly selecting: {len(data)}")
    
    if merge_ratio[i]>=1:
        data = data * int(merge_ratio[i])
        data.extend(random.sample(data, int(len(data)*(merge_ratio[i]%1))))
    else:
        if use_target and merge_ratio[i]==0:
            data = random.sample(data, min(target[i], len(data)))
        else:
            data = random.sample(data, int(len(data) * merge_ratio[i]))
    print(f"Final length after sampling: {len(data)}")
    data_list.extend(data)
print(f"Total length of merged dataset: {len(data_list)}")
json.dump(data_list, open(save_json_path, 'w'), indent=4)