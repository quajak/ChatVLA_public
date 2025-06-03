# DATA_DIR = './datasets'
import os

LOCAL_DATA_DIR = '/home/jz08/zhouzy/data'
# LOCAL_VL_IMAGE_DIR = "/home/jz08/zhouzy/data/vl_data/llava-pretrain/images"
LOCAL_VL_IMAGE_DIR = "/home/jz08/zhouzy/data/vl_data"

REMOTE_DATA_DIR = "/home/jovyan/tzb/h5py_data"
# REMOTE_VL_IMAGE_DIR = "/home/jovyan/tzb/zhumj/data/llava_finetune/"
REMOTE_VL_IMAGE_DIR = "/home/jovyan/tzb/zhouzy/data/"
DATA_DIR=REMOTE_DATA_DIR

TASK_CONFIGS = {
    "local_debug_data": {
        'dataset_dir': [
            LOCAL_DATA_DIR.replace('zhouzy','zhumj') + '/franka/4_types_pikachu_blue_van_hex_key_glove_480_640',
            LOCAL_DATA_DIR.replace('zhouzy','zhumj') + '/franka/t2',
        ],
        'vl_file': "/home/jz08/zhumj/data/vl_data/blip_laion_cc_sbu_558k.json",
        'vl_file': "/home/jz08/zhumj/data/vl_data/llava_v1_5_mix665k.json",
        'vl_image_dir': LOCAL_VL_IMAGE_DIR,
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        "sample_weights": [1, 1],
        'template_path':"/home/jz08/zhumj/data/vl_data/chat_template_vl_and_robot.json"
    },
    "local_debug_data_zzy": {
        'dataset_dir': [
            # LOCAL_DATA_DIR + '/aloha',
            # LOCAL_DATA_DIR + '/aloha_compressed',
            # LOCAL_DATA_DIR + '/franka/t1',
        ],
        # 'vl_file': "/home/jz08/zhouzy/data/vl_data/0315_math_1920_llavaocr_1783.json",
        # 'vl_file':"/home/jz08/zhouzy/data/vl_data/text_only_4000.json",
        'vl_file':"/home/jz08/zhouzy/data/vl_data/verification_code_0402.json",
        # 'vl_file':"/home/jz08/zhouzy/data/vl_data/0402_math_2k_text_2k_single_ocr_180_verification_2k.json",
        'vl_image_dir': LOCAL_VL_IMAGE_DIR,
        'episode_len': 1000,  # 1000,
        # 'camera_names': ['wrist'],
        'camera_names': ['cam_high'],
        "sample_weights": [1, 1],
        # 'template_path': "/home/jz08/zhouzy/data/vl_data/chat_template_vl_and_robot.json"
    },
    "mobile_franka_reasoning_w_vl_data": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0102_toy_blue_van_pear_tape_480_640/0102_toy_blue_van_pear_tape_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640_succ_t0001_s-0-0',# 24
            DATA_DIR + '/mobile_franka_data/0105_apple_pear_lemon_tennis_ball_480_640/0105_apple_pear_lemon_tennis_ball_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0105_brown_mug_toy_tennis_ball_sponge_480_640/0105_brown_mug_toy_tennis_ball_sponge_480_640_succ_t0001_s-0-0',# 48
            DATA_DIR + '/mobile_franka_data/0105_pink_penguin_shovel_bananan_golves_480_640/0105_pink_penguin_shovel_bananan_golves_480_640_succ_t0001_s-0-0', # 47
            DATA_DIR + '/mobile_franka_data/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640_succ_t0001_s-0-0',# 31
            ############ 500 bin picking over ########
            ##### black background start
            DATA_DIR + '/mobile_franka_data/0106_bread_to_empty_plate_480_640/0106_bread_to_empty_plate_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',# 30
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',# 32
            DATA_DIR + '/mobile_franka_data/0108_pear_to_bowl_480_640/0108_pear_to_bowl_480_640_succ_t0001_s-0-0',# 10
            DATA_DIR + '/mobile_franka_data/0108_put_pepper_to_plate_480_640/0108_put_pepper_to_plate_480_640_succ_t0001_s-0-0',# 10
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',# 20
            ##### black background over, total 127
            DATA_DIR + '/mobile_franka_data/0109_building_block_to_basket_480_640/0109_building_block_to_basket_480_640_succ_t0001_s-0-0',# 10
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',# 40
            DATA_DIR + '/mobile_franka_data/0109_pick_building_block_from_pepper_ball_mouse_sciccors_480_640/0109_pick_building_block_from_pepper_ball_mouse_sciccors_480_640_succ_t0001_s-0-0',# 20
            DATA_DIR + '/mobile_franka_data/0109_pick_pepper_from_building_block_ball_mouse_sciccors_480_640/0109_pick_pepper_from_building_block_ball_mouse_sciccors_480_640_succ_t0001_s-0-0',# 20
            ##################################### 0110_dit has above #########################
            DATA_DIR + '/mobile_franka_data/0110_green_mug_to_coaster_480_640/0110_green_mug_to_coaster_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0110_open_the_box_480_640/0110_open_the_box_480_640_succ_t0001_s-0-0',# 20
            DATA_DIR + '/mobile_franka_data/0110_pick_red_bowl_from_yellow_pepper_building_block_sponge_480_640/0110_pick_red_bowl_from_yellow_pepper_building_block_sponge_480_640_succ_t0001_s-0-0',# 21
            DATA_DIR + '/mobile_franka_data/0110_put_blue_laundry_detergent_into_box_480_640/0110_put_blue_laundry_detergent_into_box_480_640_succ_t0001_s-0-0',# 35
            DATA_DIR + '/mobile_franka_data/0110_sponge_to_pink_plate_480_640/0110_sponge_to_pink_plate_480_640_succ_t0001_s-0-0',# 20
            # # ### breakfast start ###
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_banana_to_white_plate_480_640/0111_breakfast_pick_banana_to_white_plate_480_640_succ_t0001_s-0-0',# 30
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_teaport_480_640/0111_breakfast_pick_teaport_480_640_succ_t0001_s-0-0',# 15
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_round_bread_to_green_bowl_480_640/0111_breakfast_round_bread_to_green_bowl_480_640_succ_t0001_s-0-0', # 15
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_take_a_can_of_cinnamon_powder_from_the_shelf_480_640/0111_breakfast_take_a_can_of_cinnamon_powder_from_the_shelf_480_640_succ_t0001_s-0-0', # 15
            # ### breakfast over ###
            # DATA_DIR + '/mobile_franka_data/0113_open_box_put_painter_close_box_480_640/0113_open_box_put_painter_close_box_480_640_succ_t0001_s-0-0', # 40

        ],
        'vl_file': os.path.join(REMOTE_VL_IMAGE_DIR, "llava_v1_5_mix665k.json"),
        'vl_image_dir': os.path.join(REMOTE_VL_IMAGE_DIR, "data"),
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        'template_path': os.path.join(REMOTE_VL_IMAGE_DIR,"chat_template_vl_and_robot.json")
    },
    "to_check_0216": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0114_yellow_pink_building_block_to_basket_480_640/0114_yellow_pink_building_block_to_basket_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            DATA_DIR + '/mobile_franka_data/0118_pick_orange_building_block_beside_toy_480_640/0118_pick_orange_building_block_beside_toy_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_rectangle_building_block_to_right_basket_480_640/0118_pick_up_rectangle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 50

        ],
        'vl_file': os.path.join(REMOTE_VL_IMAGE_DIR, "llava_v1_5_mix665k.json"),
        'vl_image_dir': os.path.join(REMOTE_VL_IMAGE_DIR, "data"),
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        'template_path': os.path.join(REMOTE_VL_IMAGE_DIR, "chat_template_vl_and_robot.json")
    },
    "mobile_franka_reasoning_w_vl_data_selected": {
        'dataset_dir': [
            ##### black background start
            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',# 30
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',# 20
            ##### black background over
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',# 40
            # DATA_DIR + '/mobile_franka_data/0110_green_mug_to_coaster_480_640/0110_green_mug_to_coaster_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0110_open_the_box_480_640/0110_open_the_box_480_640_succ_t0001_s-0-0',  # 20
            # DATA_DIR + '/mobile_franka_data/0113_open_box_put_painter_close_box_480_640/0113_open_box_put_painter_close_box_480_640_succ_t0001_s-0-0', # 40
            ### breakfast ###
            DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_banana_to_white_plate_480_640/0111_breakfast_pick_banana_to_white_plate_480_640_succ_t0001_s-0-0',# 30
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_teaport_480_640/0111_breakfast_pick_teaport_480_640_succ_t0001_s-0-0',# 15
            DATA_DIR + '/mobile_franka_data/0117_breakfast_pick_bread_to_plate_480_640/0117_breakfast_pick_bread_to_plate_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0117_breakfast_flip_cup_480_640/0117_breakfast_flip_cup_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0116_breakfast_take_plate_480_640/0116_breakfast_take_plate_480_640_succ_t0001_s-0-0',# 50
            ### breakfast ###
            ### toy ###
            DATA_DIR + '/mobile_franka_data/0114_yellow_pink_building_block_to_basket_480_640/0114_yellow_pink_building_block_to_basket_480_640_succ_t0001_s-0-0',# 28
            DATA_DIR + '/mobile_franka_data/0114_open_drawer_put_spider_man_480_640/0114_open_drawer_put_spider_man_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0114_stack_building_block_480_640/0114_stack_building_block_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0115_bussing_table_two_animals_two_cubes_480_640/0115_bussing_table_two_animals_two_cubes_480_640_succ_t0001_s-0-0',# 100
            DATA_DIR + '/mobile_franka_data/0118_close_drawer_480_640/0118_close_drawer_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0118_open_drawer_480_640/0118_open_drawer_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0118_pick_orange_building_block_beside_toy_480_640/0118_pick_orange_building_block_beside_toy_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_put_toy_into_drawer_480_640/0118_put_toy_into_drawer_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',  # 30
            DATA_DIR + '/mobile_franka_data/0118_pick_up_rectangle_building_block_to_right_basket_480_640/0118_pick_up_rectangle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',  # 50
            ### toy ###
            ### bathroom ###
            DATA_DIR + '/mobile_franka_data/0116_soap_to_soap_box_480_640/0116_soap_to_soap_box_480_640_succ_t0001_s-0-0',# 60
            DATA_DIR + '/mobile_franka_data/0116_bathroom_hang_cup_480_640/0116_bathroom_hang_cup_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0116_bathroom_pick_tooth_paste_480_640/0116_bathroom_pick_tooth_paste_480_640_succ_t0001_s-0-0', # 35
            DATA_DIR + '/mobile_franka_data/0116_bathroom_take_towel_480_640/0116_bathroom_take_towel_480_640_succ_t0001_s-0-0',# 30
            ### bathroom ###
            ### kitchen ###
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_pot_480_640/0116_kitchen_pick_bread_from_pot_480_640_succ_t0001_s-0-0', # 25
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_refrigerator_480_640/0116_kitchen_pick_bread_from_refrigerator_480_640_succ_t0001_s-0-0',# 20
            ### kitchen ###

        ],
        'vl_file': os.path.join(REMOTE_VL_IMAGE_DIR, "llava_v1_5_mix665k.json"),
        'vl_image_dir': os.path.join(REMOTE_VL_IMAGE_DIR, "data"),
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        'template_path': os.path.join(REMOTE_VL_IMAGE_DIR, "chat_template_vl_and_robot.json")
    },
    "mobile_franka_reasoning_no_vl_data_selected": {
        'dataset_dir': [
            ##### black background start
            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',
            # 20
            ##### black background over
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            # DATA_DIR + '/mobile_franka_data/0110_green_mug_to_coaster_480_640/0110_green_mug_to_coaster_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0110_open_the_box_480_640/0110_open_the_box_480_640_succ_t0001_s-0-0',  # 20
            # DATA_DIR + '/mobile_franka_data/0113_open_box_put_painter_close_box_480_640/0113_open_box_put_painter_close_box_480_640_succ_t0001_s-0-0', # 40
            ### breakfast ###
            DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_banana_to_white_plate_480_640/0111_breakfast_pick_banana_to_white_plate_480_640_succ_t0001_s-0-0',
            # 30
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_teaport_480_640/0111_breakfast_pick_teaport_480_640_succ_t0001_s-0-0',# 15
            DATA_DIR + '/mobile_franka_data/0117_breakfast_pick_bread_to_plate_480_640/0117_breakfast_pick_bread_to_plate_480_640_succ_t0001_s-0-0',
            # 25
            DATA_DIR + '/mobile_franka_data/0117_breakfast_flip_cup_480_640/0117_breakfast_flip_cup_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0116_breakfast_take_plate_480_640/0116_breakfast_take_plate_480_640_succ_t0001_s-0-0',
            # 50
            ### breakfast ###
            ### toy ###
            DATA_DIR + '/mobile_franka_data/0114_yellow_pink_building_block_to_basket_480_640/0114_yellow_pink_building_block_to_basket_480_640_succ_t0001_s-0-0',
            # 28
            DATA_DIR + '/mobile_franka_data/0114_open_drawer_put_spider_man_480_640/0114_open_drawer_put_spider_man_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0114_stack_building_block_480_640/0114_stack_building_block_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0115_bussing_table_two_animals_two_cubes_480_640/0115_bussing_table_two_animals_two_cubes_480_640_succ_t0001_s-0-0',
            # 100
            DATA_DIR + '/mobile_franka_data/0118_close_drawer_480_640/0118_close_drawer_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_open_drawer_480_640/0118_open_drawer_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_orange_building_block_beside_toy_480_640/0118_pick_orange_building_block_beside_toy_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_put_toy_into_drawer_480_640/0118_put_toy_into_drawer_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0118_pick_up_rectangle_building_block_to_right_basket_480_640/0118_pick_up_rectangle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 50
            ### toy ###
            ### bathroom ###
            DATA_DIR + '/mobile_franka_data/0116_soap_to_soap_box_480_640/0116_soap_to_soap_box_480_640_succ_t0001_s-0-0',
            # 60
            DATA_DIR + '/mobile_franka_data/0116_bathroom_hang_cup_480_640/0116_bathroom_hang_cup_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0116_bathroom_pick_tooth_paste_480_640/0116_bathroom_pick_tooth_paste_480_640_succ_t0001_s-0-0',
            # 35
            DATA_DIR + '/mobile_franka_data/0116_bathroom_take_towel_480_640/0116_bathroom_take_towel_480_640_succ_t0001_s-0-0',
            # 30
            ### bathroom ###
            ### kitchen ###
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_pot_480_640/0116_kitchen_pick_bread_from_pot_480_640_succ_t0001_s-0-0',
            # 25
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_refrigerator_480_640/0116_kitchen_pick_bread_from_refrigerator_480_640_succ_t0001_s-0-0',
            # 20
            ### kitchen ###

        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "mobile_franka_reasoning_w_vl_data_old": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/1228_toy_pig_yellow_pepper_pink_bus_green_plate_480_640/1228_toy_pig_yellow_pepper_pink_bus_green_plate_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/1229_mouse_clips_tinybox_bluemarkpan_480_640/1229_mouse_clips_tinybox_bluemarkpan_480_640_succ_t0001_s-0-0',
            # 31
            DATA_DIR + '/mobile_franka_data/1229_pink_cup_blue_van_bread_hex_key_480_640/1229_pink_cup_blue_van_bread_hex_key_480_640_succ_t0001_s-0-0',
            # 35
        ],
        'vl_file': os.path.join(REMOTE_VL_IMAGE_DIR, "llava_v1_5_mix665k.json"),
        'vl_image_dir': os.path.join(REMOTE_VL_IMAGE_DIR, "data"),
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
        'template_path': os.path.join(REMOTE_VL_IMAGE_DIR,"chat_template_vl_and_robot.json")
    },
    "franka_math_w_vl":{
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0', # 48
        ],
        'vl_file': [
            "/home/jovyan/tzb/zhouzy/data/processed_math_0226.json",
            "/home/jovyan/tzb/zhouzy/data/llava_v1_5_mix665k.json"
        ],
        'vl_image_dir':
            '/home/jovyan/tzb/zhouzy/data/'
        ,
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "franka_math_w_vl_llava":{
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0', # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0', # 48
        ],
        'vl_file': "/home/jovyan/tzb/zhumj/data/llava_finetune/llava_v1_5_mix665k.json",
        'vl_image_dir':'/home/jovyan/tzb/zhumj/data/llava_finetune/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "franka_math_w_vl_ocr_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file': [
            "/home/jovyan/tzb/zhouzy/data/processed_0226_and_math_algebra_counting_prealgebra.json",
            "/home/jovyan/tzb/zhouzy/data/llava_ocr.json"
        ],
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "franka_math_w_vl_text_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0226_2000_text_only_4000.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "franka_math_w_vl_single_ocr": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0226_2000_single_ocr_1800.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "franka_math_w_vl_single_ocr_left_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0226_2000_single_ocr_1800.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "franka_math_w_vl_text_only": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/text_only_4000.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0314_franka_math_w_vl_text_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0313_math_1920_text_2000.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0314_franka_math_w_vl_single_ocr": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0313_math_1920_ocr_1800.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0314_franka_math_w_vl_single_ocr_left_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0313_math_1920_ocr_1800_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0314_franka_math_w_vl_single_ocr_text_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0313_math_1920_text_2000_ocr_1980.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0315_franka_math_w_vl_llava_ocr_left_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0315_math_1920_llavaocr_1783_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0315_franka_math_w_vl_llava_ocr_text_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0315_math_1920_text_2000_llavaocr_1783.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0315_franka_math_w_vl_llava_ocr": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0315_math_1920_llavaocr_1783.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left', 'right', 'wrist'],
    },
    "0315_franka_math_w_vl_llava_ocr_text_math_left_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0315_math_1920_text_2000_llavaocr_1783_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0316_franka_math_w_vl_text_math_single_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0313_math_1920_text_2000_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0316_franka_math_w_vl_text_math_new_llava_ocr_ori_ocr_single_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0316_math_1920_text_2000_singleocr_180_newllavaocr_1783.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0317_franka_math_w_vl_text_math_three_ocr_single_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0317_math_1920_text_2000_singleocr_180_newllavaocr_1783_multiuple_ocr_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0318_franka_math_w_vl_text_math_three_ocr_single_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0318_math_4k_text_5k_three_ocr_4k_single_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0328_franka_math_w_vl_text_math_verify_ocr_wrist_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0328_math_2k_text_2k_single_ocr_180_verification_2k_wrist_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0402_franka_math_w_vl_text_math_verify_ocr_left_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0402_math_2k_text_2k_single_ocr_180_verification_2k_left_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0402_franka_math_w_vl_text_math_verify_ocr_wrist_view": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0402_math_2k_text_2k_single_ocr_180_verification_2k_wrist_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0408_test_verify_ocr": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/verification_code_0402.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0410_wrist_no_very_s1": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0410_math_6k_single_ocr_540_wrist_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0410_wrist_w_wooden_very_s2": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file':
            "/home/jovyan/tzb/zhouzy/data/0410_math_2k_text_2k_single_ocr_180_wood_verification_2k_wrist_view.json"
        ,
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0410_chatvla_for_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/llava_v1_5_mix665k.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0412_text_verify_llava": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0412_text_2025_ocr_2025_llava_2025_wrist_view.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0413_mix_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',# 50
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0413_vl_198_text_2025_ocr_2025_ori_llava_1w_wrist_view.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0420_all_robot_warmup": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
            #################################
            DATA_DIR + '/mobile_franka_data/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640/0102_green_paper_cup_yellow_bus_hex_key_gloves_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0102_toy_blue_van_pear_tape_480_640/0102_toy_blue_van_pear_tape_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640/0103_pink_penguin_lemon_cyan_trunk_gray_shovel_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640/0103_rubik_cube_apple_pink_cube_whiteboard_marker_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640/0104_rubik_cube_cyan_trunk_tape_hex_key_480_640_succ_t0001_s-0-0',# 24
            DATA_DIR + '/mobile_franka_data/0105_apple_pear_lemon_tennis_ball_480_640/0105_apple_pear_lemon_tennis_ball_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0105_brown_mug_toy_tennis_ball_sponge_480_640/0105_brown_mug_toy_tennis_ball_sponge_480_640_succ_t0001_s-0-0',# 48
            DATA_DIR + '/mobile_franka_data/0105_pink_penguin_shovel_bananan_golves_480_640/0105_pink_penguin_shovel_bananan_golves_480_640_succ_t0001_s-0-0', # 47
            DATA_DIR + '/mobile_franka_data/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640/0105_green_paper_cup_cutter_knife_whiteboard_marker_brown_plate_480_640_succ_t0001_s-0-0',# 31
            ############ 500 bin picking over ########
            ##### black background start
            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',  # 20
            ##### black background over
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            # DATA_DIR + '/mobile_franka_data/0110_green_mug_to_coaster_480_640/0110_green_mug_to_coaster_480_640_succ_t0001_s-0-0',# 25
            DATA_DIR + '/mobile_franka_data/0110_open_the_box_480_640/0110_open_the_box_480_640_succ_t0001_s-0-0',  # 20
            # DATA_DIR + '/mobile_franka_data/0113_open_box_put_painter_close_box_480_640/0113_open_box_put_painter_close_box_480_640_succ_t0001_s-0-0', # 40
            ### breakfast ###
            DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_banana_to_white_plate_480_640/0111_breakfast_pick_banana_to_white_plate_480_640_succ_t0001_s-0-0',
            # 30
            # DATA_DIR + '/mobile_franka_data/0111_breakfast_pick_teaport_480_640/0111_breakfast_pick_teaport_480_640_succ_t0001_s-0-0',# 15
            DATA_DIR + '/mobile_franka_data/0117_breakfast_pick_bread_to_plate_480_640/0117_breakfast_pick_bread_to_plate_480_640_succ_t0001_s-0-0',
            # 25
            DATA_DIR + '/mobile_franka_data/0117_breakfast_flip_cup_480_640/0117_breakfast_flip_cup_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0116_breakfast_take_plate_480_640/0116_breakfast_take_plate_480_640_succ_t0001_s-0-0',
            # 50
            ### breakfast ###
            ### toy ###
            DATA_DIR + '/mobile_franka_data/0114_yellow_pink_building_block_to_basket_480_640/0114_yellow_pink_building_block_to_basket_480_640_succ_t0001_s-0-0',
            # 28
            DATA_DIR + '/mobile_franka_data/0114_open_drawer_put_spider_man_480_640/0114_open_drawer_put_spider_man_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0114_stack_building_block_480_640/0114_stack_building_block_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0115_bussing_table_two_animals_two_cubes_480_640/0115_bussing_table_two_animals_two_cubes_480_640_succ_t0001_s-0-0',
            # 100
            DATA_DIR + '/mobile_franka_data/0118_close_drawer_480_640/0118_close_drawer_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_open_drawer_480_640/0118_open_drawer_480_640_succ_t0001_s-0-0',  # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_orange_building_block_beside_toy_480_640/0118_pick_orange_building_block_beside_toy_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_put_toy_into_drawer_480_640/0118_put_toy_into_drawer_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0118_pick_up_rectangle_building_block_to_right_basket_480_640/0118_pick_up_rectangle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 50
            ### toy ###
            ### bathroom ###
            DATA_DIR + '/mobile_franka_data/0116_soap_to_soap_box_480_640/0116_soap_to_soap_box_480_640_succ_t0001_s-0-0',  # 60
            DATA_DIR + '/mobile_franka_data/0116_bathroom_hang_cup_480_640/0116_bathroom_hang_cup_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0116_bathroom_pick_tooth_paste_480_640/0116_bathroom_pick_tooth_paste_480_640_succ_t0001_s-0-0',
            # 35
            DATA_DIR + '/mobile_franka_data/0116_bathroom_take_towel_480_640/0116_bathroom_take_towel_480_640_succ_t0001_s-0-0',
            # 30
            ### bathroom ###
            ### kitchen ###
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_pot_480_640/0116_kitchen_pick_bread_from_pot_480_640_succ_t0001_s-0-0',
            # 25
            DATA_DIR + '/mobile_franka_data/0116_kitchen_pick_bread_from_refrigerator_480_640/0116_kitchen_pick_bread_from_refrigerator_480_640_succ_t0001_s-0-0',
            # 20
            ### kitchen ###
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0413_vl_198_text_2025_ocr_2025_ori_llava_1w_wrist_view.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0420_llava_warmup": {
        'dataset_dir': [
            # DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/llava_v1_5_mix665k.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0424_mix_no_ocr_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',# 50
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0424_vl_198_text_2025_ori_llava_1w_wrist_view.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0424_action_post": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0103_brown_mug_cutter_knife_bread_banana_480_640/0103_brown_mug_cutter_knife_bread_banana_480_640_succ_t0001_s-0-0',# 50
            DATA_DIR + '/mobile_franka_data/0103_green_can_tennis_ball_sponge_brown_plate_480_640/0103_green_can_tennis_ball_sponge_brown_plate_480_640_succ_t0001_s-0-0',# 50
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0426_mix_no_ocr_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48

            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',# 30
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',  # 20
            DATA_DIR + '/mobile_franka_data/0118_put_toy_into_drawer_480_640/0118_put_toy_into_drawer_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0426_text_2025_filter_llava_1w_processed.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0426_action_post": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640/0224_1_plus_7_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640/0224_2_plus_5_480_640_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640/0224_3_plus_6_480_640_succ_t0001_s-0-0',  # 48

            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',
            # 20
            DATA_DIR + '/mobile_franka_data/0118_put_toy_into_drawer_480_640/0118_put_toy_into_drawer_480_640_succ_t0001_s-0-0',
            # 50
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0428_action_post": { ### cut half of action!
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640_modified/0224_1_plus_4_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640_modified/0224_1_plus_7_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640_modified/0224_2_plus_5_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640_modified/0224_3_plus_6_480_640_modified_succ_t0001_s-0-0',  # 48

            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',
            # 20
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0428_left_mix_no_ocr_math": {
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640_modified/0224_1_plus_4_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640_modified/0224_1_plus_7_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640_modified/0224_2_plus_5_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640_modified/0224_3_plus_6_480_640_modified_succ_t0001_s-0-0',  # 48

            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',
            # 20
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0426_text_2025_filter_llava_1w_processed.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0430_action_post": { ### cut half of action!
        'dataset_dir': [
            DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640_modified/0224_1_plus_4_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_1_plus_7_480_640_modified/0224_1_plus_7_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_2_plus_5_480_640_modified/0224_2_plus_5_480_640_modified_succ_t0001_s-0-0',  # 48
            DATA_DIR + '/mobile_franka_data/0224_3_plus_6_480_640_modified/0224_3_plus_6_480_640_modified_succ_t0001_s-0-0',  # 48

            DATA_DIR + '/mobile_franka_data/0107_bread_to_empty_plate_spoon_480_640/0107_bread_to_empty_plate_spoon_480_640_succ_t0001_s-0-0',
            # 30
            DATA_DIR + '/mobile_franka_data/0109_green_cube_to_pink_cube_480_640/0109_green_cube_to_pink_cube_480_640_succ_t0001_s-0-0',
            # 40
            DATA_DIR + '/mobile_franka_data/0108_hang_cup_480_640/0108_hang_cup_480_640_succ_t0001_s-0-0',  # 32
            DATA_DIR + '/mobile_franka_data/0108_put_tennis_ball_480_640/0108_put_tennis_ball_480_640_succ_t0001_s-0-0',
            # 20
            DATA_DIR + '/mobile_franka_data/0118_pick_up_semicircle_building_block_to_right_basket_480_640/0118_pick_up_semicircle_building_block_to_right_basket_480_640_succ_t0001_s-0-0',
            # 30
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['left'],
    },
    "0428_aloha": {
        'dataset_dir': [
            DATA_DIR + '/aloha_quest3_data/full_data/62853',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/54901',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/25780',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/13469',  # 85
            DATA_DIR + '/aloha_quest3_data/compressed_data/03312',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/10175',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/33639',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/41542',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/64106',#90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/48128',#90
            #### ori aloha
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0426_text_2025_filter_llava_1w_processed.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['cam_high'],
    },
    "0430_aloha_action_post": {
        'dataset_dir': [
            DATA_DIR + '/aloha_quest3_data/full_data/62853',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/54901',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/25780',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/13469',  # 85
            DATA_DIR + '/aloha_quest3_data/compressed_data/03312',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/10175',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/33639',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/41542',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/64106',#90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/48128',#90
            #### ori aloha
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['cam_high'],
    },
    "0501_aloha": {
        'dataset_dir': [
            DATA_DIR + '/aloha_quest3_data/full_data/62853',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/54901',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/25780',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/13469',  # 85
            DATA_DIR + '/aloha_quest3_data/compressed_data/03312',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/10175',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/33639',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/41542',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/64106',#90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/48128',#90
            #### ori aloha
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0426_text_2025_filter_llava_1w_processed.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['cam_high'],
    },
    "0503_aloha": {
        'dataset_dir': [
            DATA_DIR + '/aloha_quest3_data/full_data/62853',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/54901',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/25780',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/13469',  # 85
            DATA_DIR + '/aloha_quest3_data/compressed_data/03312',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/10175',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/33639',#60
            DATA_DIR + '/aloha_quest3_data/compressed_data/41542',#60
            ##### 600
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/64106',#90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/48128',#90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0428_77148',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0428_97163',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0430_147218',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0429_corridor_99184_lyp',  # 90
            ##### 540
            #### ori aloha
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0503_cutted_llava.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['cam_high'],
    },
    "0503_llava_warmup": {
        'dataset_dir': [
            # DATA_DIR + '/mobile_franka_data/0224_1_plus_4_480_640/0224_1_plus_4_480_640_succ_t0001_s-0-0',  # 48
        ],
        'vl_file': "/home/jovyan/tzb/zhouzy/data/0503_cutted_llava.json",
        'vl_image_dir': '/home/jovyan/tzb/zhouzy/data/',
        'episode_len': 1000,  # 1000,
        'camera_names': ['wrist'],
    },
    "0503_aloha_post": {
        'dataset_dir': [
            DATA_DIR + '/aloha_quest3_data/full_data/62853',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/54901',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/25780',  # 90
            DATA_DIR + '/aloha_quest3_data/full_data/13469',  # 85
            DATA_DIR + '/aloha_quest3_data/compressed_data/03312',  # 60
            DATA_DIR + '/aloha_quest3_data/compressed_data/10175',  # 60
            DATA_DIR + '/aloha_quest3_data/compressed_data/33639',  # 60
            DATA_DIR + '/aloha_quest3_data/compressed_data/41542',  # 60
            ##### 600
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/64106',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/48128',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0428_77148',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0428_97163',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0430_147218',  # 90
            DATA_DIR + '/aloha_quest3_data/compressed_data/pedal/0429_corridor_99184_lyp',  # 90
            ##### 540
            #### ori aloha
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/aloha_data/hang_cups_waibao',
            '/home/jovyan/tzb/h5py_data/aloha_bimanual/aloha_4views/pour_rice_yichen_0102',
        ],
        'episode_len': 1000,  # 1000,
        'camera_names': ['cam_high'],
    },



#     0413_vl_198_text_2025_ocr_2025_ori_llava_1w
# 0426_text_2025_filter_llava_1w_processed.json
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
FPS = 50
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
