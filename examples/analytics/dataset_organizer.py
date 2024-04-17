import fog_x




DATASETS = [
    "fractal20220817_data",
    "kuka",
    "bridge",
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds",
    "nyu_rot_dataset_converted_externally_to_rlds",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "maniskill_dataset_converted_externally_to_rlds",
    "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "usc_cloth_sim_converted_externally_to_rlds",
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds",
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds",
    "utokyo_saytap_converted_externally_to_rlds",
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "robo_net",
    "berkeley_mvp_converted_externally_to_rlds",
    "berkeley_rpt_converted_externally_to_rlds",
    "kaist_nonprehensile_converted_externally_to_rlds",
    "stanford_mask_vit_converted_externally_to_rlds",
    "tokyo_u_lsmo_converted_externally_to_rlds",
    "dlr_sara_pour_converted_externally_to_rlds",
    "dlr_sara_grid_clamp_converted_externally_to_rlds",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "asu_table_top_converted_externally_to_rlds",
    "stanford_robocook_converted_externally_to_rlds",
    "eth_agent_affordances",
    "imperialcollege_sawyer_wrist_cam",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "uiuc_d3field",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_play_fusion",
    "cmu_stretch",
    "berkeley_gnm_recon",
    "berkeley_gnm_cory_hall",
    # "berkeley_gnm_sac_son",
]


objects = ["marker", "cloth", "cup", "object", "bottle", "block", "drawer", "lid", "mug"]
tasks = ["put", "move", "pick", "remove", "take", "open", "close", "place", "turn", "push", 
         "insert", "stack", "lift", "pour"] # things not in DROID
views = ["wrist", "top", "other"]

dataset_id = 0
for dataset_name in DATASETS:
    dataset = fog_x.dataset.Dataset(
        name=dataset_name,
        path="~/test_dataset",
    )

    # dataset._prepare_rtx_metadata(
    #     name=dataset_name,
    #     sample_size = 10,
    #     shuffle=True,
    # )

    info = dataset.get_episode_info()

    for episode_metadata in info.iter_rows(named = True):
        instruction = episode_metadata["natural_language_instruction"]

        d = dict()
        instruction = instruction.lower().replace(",", "").replace("\n", "")
        d["dataset_id"] = f"dataset-{dataset_id}"
        d["info"] = instruction
        task_id = -1 
        for task in tasks:
            if task in instruction:
                task_id = tasks.index(task)
        if task_id == -1:
            task_id = len(tasks) - 1

        obj_id = -1
        for obj in objects:
            if obj in instruction:
                obj_id = objects.index(obj)
        if obj_id == -1:
            obj_id = len(objects) - 1
            
        d["task_id"] = f"task-{task_id}"
        d["object_id"] = f"object-{obj_id}"

        images_features = [col for col in info.columns if col.startswith("video_path_")]
        for i, image_feature in enumerate(images_features):
            path = episode_metadata[image_feature]
            d["poster"] = f"videos/{dataset_name}_viz/{path}.jpg"
            d["src"] = f"videos/{dataset_name}_viz/{path}.mp4"
            view_id = -1
            for view in views:
                if view in path:
                    view_id = views.index(view)
                    break
            if view_id == -1:   
                view_id = len(views) - 1
            
            d["view_id"] = f"view-{view_id}"

            # print d in JSON format 
            with open("/tmp/dataset_info.txt", "a") as file:
                printable = str(d).replace("\'", "\"")
                file.write(f'JSON.parse(\'{printable}\'),\n')

            
        # write as a line of JSON.parse('{"info": "Unfold the tea towel", "poster": "videos/bridge_viz/bridge_0_image.jpg", "src": "videos/bridge_viz/bridge_0_image.mp4"}'),
        # print (f'JSON.parse(\'{{"info": "{instruction}", "poster": "videos/{dataset_name}_viz/{dataset_name}_{episode_id}_image.jpg", "src": "videos/{dataset_name}_viz/{dataset_name}_{dataset_id}_image.mp4"}}\'),')
    dataset_id += 1