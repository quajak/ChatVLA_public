import torch

from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
import numpy as np
CAMERA_VIEWS=['cam_bottom', 'cam_top', 'cam_left_wrist', 'cam_right_wrist']

from qwen2_vla.model_load_utils import load_model_for_eval
class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]
        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(
            model_path=model_path,
            model_base=model_base, policy_config=policy_config)

        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def datastruct_droid2qwen2vla(self, raw_lang):
        messages = [
            {
                "role": "user",
                "content": [
                ],
            },
        ]
        for i in range(len(self.policy_config["camera_views"])):
            messages[0]['content'].append(
                {
                    "type": "image",
                    "image": None,
                }
            )
        messages[0]['content'].append({"type": "text", "text": f""})
        messages[0]['content'][-1]['text'] = raw_lang
        return messages

    def process_batch_to_vla(self, curr_image, robo_state, raw_lang):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # left, right ,wrist for franka
        image_list = []
        for i, each in enumerate(image_data):
            ele = {
            }
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            ele['resized_height'] = 240
            ele['resized_width'] = 320
            each = fetch_image(ele)
            image_list.append(torch.from_numpy(np.array(each)))

        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict