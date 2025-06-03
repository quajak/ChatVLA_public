import os
import time
import json
from os.path import isdir
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import h5py
import numpy as np
from tqdm import tqdm
from scipy import signal


class AlohaDataProcessor:
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.reasoning_file = Path('./data_utils/saved_reasoning_json')
        self.reasoning_file.mkdir(exist_ok=True, parents=True)
        self.reasoning_file = self.reasoning_file / 'math_aloha_0428.json'
        
    def load_task_config(self, config_file: Union[str, Path]) -> Dict:
        """Load task configuration from a JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Dictionary containing task configuration
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        try:
            with open(config_path, 'r') as f:
                task_config = json.load(f)
            print(f"Loaded task configuration for {len(task_config)} tasks from {config_file}")
            return task_config
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing config file {config_file}: {e}")

    def generate_reasoning_from_templates(self, task_config: Dict[str, Dict[str, Any]]) -> Dict:
        save_reasoning_data = {}
        
        for task_id, task in enumerate(task_config):
            print(f"{'<'*20}processing task {task}{'>'*20}")
            config = task_config[task]
            with open(config['template_path'], 'r') as f:
                task_template = json.load(f)[config['reasoning_type']]

            num_episodes = 6 * config['num_trial_each_group']
            file_list = [f'episode_{i}.hdf5' for i in range(num_episodes)]
            save_reasoning_data[task] = {}

            for file_id, file_name in tqdm(enumerate(file_list)):
                sub_reasoning = self._process_reasoning_template(
                    task_template=task_template.copy(),
                    config=config,
                    file_id=file_id
                )
                
                # Debug print like in original code
                group_id = file_id // config['num_trial_each_group']
                print(file_id, group_id, sub_reasoning)
                
                try:
                    save_reasoning_data[task][file_name] = sub_reasoning
                except Exception as e:
                    print(f"Error processing {task}/{file_name}: {e}")
                    return save_reasoning_data
                    
        # Save reasoning data, merging with existing data if present
        self._save_reasoning_data(save_reasoning_data)
        return save_reasoning_data
    
    def _process_reasoning_template(self, task_template: List[str], 
                                   config: Dict[str, Any], 
                                   file_id: int) -> List[str]:
        group_id = file_id // config['num_trial_each_group']
        permutations = [
            [0, 1, 2], [0, 2, 1], [2, 0, 1], 
            [1, 0, 2], [1, 2, 0], [2, 1, 0]
        ]
        
        sub_reasoning = task_template.copy()
        for idx in range(len(sub_reasoning)):
            # Replace static placeholders
            for placeholder, value in config['placeholders'].items():
                sub_reasoning[idx] = sub_reasoning[idx].replace(str(placeholder), str(value))
            
            # Replace position placeholders
            pos_placeholders = {
                '<pos1>': config['roll_placeholders']['pos'][group_id // 2],
                '<pos2>': config['roll_placeholders']['pos'][((group_id + 3) % 6) // 2],
                '<num1>': config['roll_placeholders']['answer'][0],
                '<num2>': config['roll_placeholders']['answer'][1],
                '<ans1>': config['roll_placeholders']['answer'][permutations[group_id][0]],
                '<ans2>': config['roll_placeholders']['answer'][permutations[group_id][1]],
                '<ans3>': config['roll_placeholders']['answer'][permutations[group_id][2]]
            }
            
            for placeholder, value in pos_placeholders.items():
                sub_reasoning[idx] = sub_reasoning[idx].replace(placeholder, str(value))
                
        return sub_reasoning
    
    def _save_reasoning_data(self, save_reasoning_data: Dict):
        if self.reasoning_file.exists():
            try:
                with open(self.reasoning_file, 'r') as f:
                    existing_data = json.load(f)

                for task in save_reasoning_data:
                    existing_data[task] = save_reasoning_data[task]
                save_reasoning_data = existing_data
            except json.JSONDecodeError:
                print(f"Error reading {self.reasoning_file}, starting with fresh data")
        
        with open(self.reasoning_file, 'w') as f:
            json.dump(save_reasoning_data, f, indent=4)
    
    def add_subtask_to_hdf5(self, file_path: Union[str, Path], sub_reasonings: List[str]):
        with h5py.File(file_path, 'a') as hdf5_file:
            data_len = hdf5_file['action'][()].shape[0]

            if "subtask" in hdf5_file:
                sub_index_ori = hdf5_file['subtask'][()]
                if sub_index_ori.shape[0] > data_len:
                    sub_index_ori = sub_index_ori[:-1]
                sub_index_ori = np.array([int(i[0] == 10) for i in sub_index_ori])
                substep_index = np.cumsum(np.concatenate(([0], np.diff(sub_index_ori) > 0)))
            else:
                substep_index = np.zeros(data_len, dtype=int)

            if sub_reasonings is not None:
                targets = [sub_reasonings[i] for i in substep_index]

                if "substep_reasonings" in hdf5_file:
                    del hdf5_file['substep_reasonings']
                hdf5_file.create_dataset("substep_reasonings", data=targets)
                print(targets[0], targets[-1])

    def add_reasoning_to_hdf5_files(self, data_folder: Union[str, Path],
                                    task_list: List[str],
                                    reasoning_data: Dict,
                                    exclude_list: List[str] = None):
        data_folder = Path(data_folder)
        exclude_list = exclude_list or []

        for task in task_list:
            if task in exclude_list or any(exclude_key in task for exclude_key in exclude_list):
                continue

            if task not in reasoning_data:
                print(f"No reasoning data found for task {task}, skipping")
                continue

            print(f"{'<' * 10} Adding reasoning for task {task} {'>' * 10}")
            task_path = data_folder / task

            h5_data_list = sorted([f for f in os.listdir(task_path) if f.endswith('.hdf5')])

            # Process each file
            for idx, h5_data in tqdm(enumerate(h5_data_list)):
                file_path = task_path / h5_data

                try:
                    if h5_data in reasoning_data[task]:
                        sub_reasonings = reasoning_data[task][h5_data]
                        self.add_subtask_to_hdf5(file_path, sub_reasonings)
                    else:
                        print(f"No reasoning data found for file {h5_data}, skipping")
                except Exception as e:
                    print(f"Error adding reasoning to {file_path}: {e}")

            # Pause between tasks to allow system to recover
            time.sleep(1)

    def prepare_single_hdf5_file(self, file_path: Union[str, Path], raw_lang: str = ""):

        with h5py.File(file_path, 'a') as hdf5_file:
            # Generate qpos and action if not present
            if 'action' not in hdf5_file:
                left_qpos = hdf5_file['/state/joint_position/left'][()]
                right_qpos = hdf5_file['/state/joint_position/right'][()]
                qpos = np.concatenate([left_qpos, right_qpos], axis=-1)
                data_len = qpos.shape[0] - 1
                
                # Create missing datasets
                if '/observations/qpos' not in hdf5_file:
                    hdf5_file.create_dataset('/observations/qpos', data=qpos[:-1])
                
                if '/observations/qvel' not in hdf5_file:
                    hdf5_file.create_dataset('/observations/qvel', data=np.zeros_like(qpos[:-1]))
                
                hdf5_file.create_dataset('action', data=qpos[1:])

                for key in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
                    if key in hdf5_file['observations']['images']:
                        sample_data = hdf5_file['observations']['images'][key][:]
                        if sample_data.shape[0] > data_len:
                            sample_data = sample_data[:-1]
                            del hdf5_file['observations']['images'][key]
                            hdf5_file.create_dataset(f'/observations/images/{key}', data=sample_data)

            if raw_lang:
                if "language_raw" in hdf5_file:
                    del hdf5_file["language_raw"]
                hdf5_file.create_dataset("language_raw", data=[raw_lang])

    def prepare_hdf5_files(self, data_folder: Union[str, Path],
                           task_list: List[str],
                           exclude_list: List[str] = None,
                           raw_lang: str = "Answer the question and pick the card with the correct answer."):
        data_folder = Path(data_folder)
        exclude_list = exclude_list or []

        for task in task_list:
            if task in exclude_list or any(exclude_key in task for exclude_key in exclude_list):
                continue

            print(f"{'<' * 10} Preparing files for task {task} {'>' * 10}")
            task_path = data_folder / task

            h5_data_list = sorted([f for f in os.listdir(task_path) if f.endswith('.hdf5')])

            for idx, h5_data in tqdm(enumerate(h5_data_list)):
                file_path = task_path / h5_data

                try:
                    self.prepare_single_hdf5_file(file_path, raw_lang=raw_lang)
                except Exception as e:
                    print(f"Error preparing {file_path}: {e}")

            time.sleep(1)

    def rename_hdf5_files(self, data_folder: Union[str, Path],
                        task_list: List[str], 
                        exclude_list: List[str] = None):

        data_folder = Path(data_folder)
        exclude_list = exclude_list or []
        
        for task in task_list:
            if task in exclude_list or any(exclude_key in task for exclude_key in exclude_list):
                continue
                
            print(f"{'<'*10} Renaming files for task {task} {'>'*10}")
            task_path = data_folder / task
            
            h5_data_list = sorted([f for f in os.listdir(task_path) if f.endswith('.hdf5')])
            
            # Process each file
            for idx, h5_data in tqdm(enumerate(h5_data_list)):
                file_path = task_path / h5_data
                new_filename = f"episode_{idx}.hdf5"
                new_path = task_path / new_filename
                
                try:
                    print(f"Renaming {h5_data} to {new_filename}")
                    if new_path != file_path:  # Avoid renaming if already in correct format
                        os.rename(file_path, new_path)
                except Exception as e:
                    print(f"Error renaming {file_path}: {e}")
    
    def check_single_file(self, file_path: Union[str, Path], plot: bool = False) -> Dict[str, Any]:
        """Check the action data in a single HDF5 file and analyze if smoothing is needed.
        
        Args:
            file_path: Path to the HDF5 file
            plot: Whether to generate a plot of the action data if matplotlib is available
            
        Returns:
            Dictionary with statistics about the action data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with h5py.File(file_path, 'r') as hdf5_file:
            if 'action' not in hdf5_file:
                return {"error": "No action data found in this file"}
            
            # Get action data
            action_data = hdf5_file['action'][()]
            
            # Calculate basic statistics
            stats = {
                "shape": action_data.shape,
                "mean": float(np.mean(action_data)),
                "std": float(np.std(action_data)),
                "min": float(np.min(action_data)),
                "max": float(np.max(action_data)),
                "timesteps": action_data.shape[0]
            }
            
            # Calculate first derivative (velocity) to check for jumps
            action_diff = np.diff(action_data, axis=0)
            max_diff = np.max(np.abs(action_diff), axis=0)
            mean_diff = np.mean(np.abs(action_diff), axis=0)
            
            stats["max_jump"] = float(np.max(max_diff))
            stats["mean_jump"] = float(np.mean(mean_diff))

            return stats

def main():

    data_folder = "/home/jovyan/tzb/h5py_data/aloha_quest3_data/compressed_data/pedal/"
    data_folder = "/media/jz08/HDD/aloha_quest3_data/pedal/0427_test"
    exclude_list = []
    task_list = os.listdir(data_folder)
    task_list = [t for t in task_list if isdir(os.path.join(data_folder,t))]
    task_list.sort()
    # task_list = ['0429_corridor_99184_lyp']
    template_path = "/home/jz08/zhouzy/code/moevla/data_utils/generate_reasoning/using_reasoning_same_aloha.json"
    raw_lang = "Answer the question and pick the card with the correct answer."
    # existing_data_json = '/home/jovyan/tzb/zhouzy/code/moevla/data_utils/saved_data_reasoning_json/math_aloha_0428.json'
    existing_data_json = '/home/jz08/zhouzy/code/moevla/data_utils/saved_reasoning_json/math_aloha_0428.json'
    
    # Path to task configuration file
    task_config_file = '/data_utils/task_config/math_aloha_task_config.json'

    processor = AlohaDataProcessor()
    
    # Choose which operations to perform
    do_rename = False
    do_generate_reasoning = False
    do_prepare = False
    do_add_reasoning = True
    do_check_action = False

    
    if do_rename:
        print("\n=== STEP 1: RENAMING HDF5 FILES ===")
        processor.rename_hdf5_files(
            data_folder=data_folder,
            task_list=task_list,
            exclude_list=exclude_list
        )
    

    reasoning_data = None
    if do_generate_reasoning:
        print("\n=== STEP 2: GENERATING REASONING DATA ===")
        try:
            # Load task configuration from JSON file
            task_config = processor.load_task_config(task_config_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading task config: {e}")
            print("Using default task configuration instead.")
            # Fallback to default configuration
            task_config = {
            }
        
        # Save the default configuration if it doesn't exist yet
        if not Path(task_config_file).exists():
            print(f"Creating task configuration file: {task_config_file}")
            with open(task_config_file, 'w') as f:
                json.dump(task_config, f, indent=4)
        
        reasoning_data = processor.generate_reasoning_from_templates(task_config)

    if do_prepare:
        print("\n=== STEP 3: PREPARING HDF5 FILES ===")
        processor.prepare_hdf5_files(
            data_folder=data_folder,
            task_list=task_list,
            exclude_list=exclude_list,
            raw_lang=raw_lang
        )

    if do_add_reasoning:
        print("\n=== STEP 4: ADDING REASONING TO HDF5 FILES ===")
        with open(existing_data_json, 'r') as f:
            reasoning_data = json.load(f)

        processor.add_reasoning_to_hdf5_files(
            data_folder=data_folder,
            task_list=task_list,
            reasoning_data=reasoning_data,
            exclude_list=exclude_list
        )
    
    if do_check_action:
        print("\n=== STEP 5: CHECKING ACTION DATA ===")
        # Example of checking a single file
        sample_task = task_list[0]
        sample_file = os.path.join(data_folder, sample_task, "episode_1.hdf5")
        
        if os.path.exists(sample_file):
            print(f"Checking action data in {sample_file}")
            stats = processor.check_single_file(sample_file, plot=True)
            print("Action statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print(f"Sample file {sample_file} not found.")

    
    print("\n=== ALL PROCESSING COMPLETED ===")


if __name__ == "__main__":
    main()
