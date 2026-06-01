from dexbotic.data.data_source.register import register_dataset


POST_SORT_DM0_NOBFRAMES_DATASET = {
    "data_merged_dm0_dexdata_nobframes_train": {
        "data_path_prefix": "/dexbotic/data/post_data_merged_dm0_dexdata_nobframes_train/video",
        "annotations": "/dexbotic/data/post_data_merged_dm0_dexdata_nobframes_train/jsonl",
        "frequency": 1,
    },
    "data_merged_dm0_dexdata_nobframes_test": {
        "data_path_prefix": "/dexbotic/data/post_data_merged_dm0_dexdata_nobframes_test/video",
        "annotations": "/dexbotic/data/post_data_merged_dm0_dexdata_nobframes_test/jsonl",
        "frequency": 1,
    },
    "origin_data_0423_dm0_dexdata_nobframes_train": {
        "data_path_prefix": "/dexbotic/data/post_origin_data_0423_dm0_dexdata_nobframes_train/video",
        "annotations": "/dexbotic/data/post_origin_data_0423_dm0_dexdata_nobframes_train/jsonl",
        "frequency": 1,
    },
    "origin_data_0423_dm0_dexdata_nobframes_test": {
        "data_path_prefix": "/dexbotic/data/post_origin_data_0423_dm0_dexdata_nobframes_test/video",
        "annotations": "/dexbotic/data/post_origin_data_0423_dm0_dexdata_nobframes_test/jsonl",
        "frequency": 1,
    },
}

meta_data = {
    "non_delta_mask": [6, 13],
    "periodic_mask": None,
    "periodic_range": None,
}

register_dataset(
    POST_SORT_DM0_NOBFRAMES_DATASET,
    meta_data=meta_data,
    prefix="post",
)
