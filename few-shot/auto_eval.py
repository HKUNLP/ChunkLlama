import os
import argparse

DATASETS = [
    "narrative_qa",
    "qasper",
    "summ_screen_fd",
    "gov_report",
    "qmsum",
    "contract_nli",
    "quality",
    "quality_hard",
]


def main(args, raise_on_errors=False):
    """
    If raise_on_errors is True, raises ValueError on verification errors (after dumping the error descriptions).
    Otherwise, exists with an error code
    """
    predictions = args.predictions
    dataset_name = args.dataset_name
    verify_only = args.verify_only


    from scrolls_metrics.scrolls import  Scrolls
    scrolls_metric = Scrolls(config_name=dataset_name)
    # Downloading and loading the dataset from the hub
    load_dataset_kwargs = {
        "path": "scrolls_metrics/scrolls.py",
        "name": dataset_name if dataset_name != "quality_hard" else "quality",
        "hard_only": None if dataset_name != "quality_hard" else True,
        "data_files": {"test": args.test_data_file} if args.test_data_file is not None else None,
    }
    if args.cache_dir is not None:
        load_dataset_kwargs["cache_dir"] = args.cache_dir
    load_dataset_kwargs["split"] = args.split

    # seq2seq_dataset = load_dataset(**load_dataset_kwargs)
    import json
    from datasets import Dataset

    # 手动读取JSONL文件
    with open(args.test_data_file, 'r', encoding='utf-8') as json_file:
        data = [json.loads(line) for line in json_file]
    data_dict = {key: [dic[key] for dic in data] for key in data[0]}
    # 将数据转换为 Dataset 对象
    seq2seq_dataset = Dataset.from_dict(data_dict)
    if not verify_only:
        assert all(
            example["output"] is not None for example in seq2seq_dataset
        ), "Make sure to load data with gold outputs"

    # Prepare reference
    untokenized_dataset = drop_duplicates_in_input(seq2seq_dataset)
    id_to_labels = {instance["id"]: instance["outputs"] for instance in untokenized_dataset}

    # Prepare predictions
    if isinstance(predictions, str):
        with open(predictions) as f:
            id_to_pred = json.load(f)
    else:
        id_to_pred = predictions

    # Check for format errors
    errors, details = verify(id_to_pred, id_to_labels)

    out_file_path = get_metrics_filename(args.metrics_output_dir, dataset_name)
    os.makedirs(args.metrics_output_dir, exist_ok=True)

    if len(errors) == 0 and not verify_only:
        # Compute metrics
        metrics = scrolls_metric.compute(**scrolls_metric.convert_from_map_format(id_to_pred, id_to_labels))

        with open(out_file_path, mode="w") as f:
            json.dump(metrics, f, indent=4)

        if args.internal_call:
            return metrics
        else:
            print(json.dumps(metrics, indent=4))
    elif len(errors) > 0:
        # Output errors
        errors_msg = errors[0] if len(errors) == 1 else " ".join(f"{i}: {err}" for i, err in enumerate(errors))
        print(json.dumps(errors, indent=4))
        print(f"See details in: {out_file_path}")
        with open(out_file_path, mode="w") as f:
            json.dump({"errors": errors, "details": details}, f, indent=4)
        if raise_on_errors:
            raise ValueError(f"Failed to evaluate due to: {errors_msg}")
        exit(os.EX_DATAERR)


# def download_metric():
#     scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset")
#     updated_scrolls_metric_path = (
#         os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
#     )
#     shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
#     return updated_scrolls_metric_path


# Copied from baselines/src/utils/duplicates.py
def drop_duplicates_in_input(untokenized_dataset):
    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(zip(untokenized_dataset["id"], untokenized_dataset["output"])):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def get_metrics_filename(outdir, dataset_name):
    return os.path.join(outdir, f"{dataset_name}_metrics.json")


def verify(id_to_pred, id_to_labels):
    errors = []
    details = {"missing_keys": [], "redundant_keys": []}
    if not isinstance(id_to_pred, dict):
        errors.append('The predictions must be saved a JSON object: {"id1": "prediction1", "id2": "prediction2", ...}')
    else:
        if not all(isinstance(key, str) for key in id_to_pred.keys()):
            errors.append("All keys of the predictions dictionary must be strings")
        if not all(isinstance(value, str) for value in id_to_pred.values()):
            errors.append("All values of the predictions dictionary must be strings")
        if len(errors) == 0:
            predictions_keys, reference_keys = set(id_to_pred.keys()), set(id_to_labels.keys())
            missing_keys = reference_keys - predictions_keys
            redundant_keys = predictions_keys - reference_keys

            if len(missing_keys) > 0:
                details["missing_keys"] = list(missing_keys)
                # errors.append(f"There are missing example IDs.")
            else:
                del details["missing_keys"]

            if len(redundant_keys) > 0:
                details["redundant_keys"] = list(redundant_keys)
                errors.append(f"There are redundant example IDs.")
            else:
                del details["redundant_keys"]

    return errors, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SCROLLS predictions per dataset")
    parser.add_argument(
        "--predictions", type=str, help="Path to the predictions file or the actual predictions", required=True
    )
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset", choices=DATASETS, required=True)
    parser.add_argument("--metrics_output_dir", type=str, help="Directory of the output metrics file", required=True)
    parser.add_argument("--split", type=str, help="The split to evaluate on", default="test")
    parser.add_argument("--internal_call", type=str, help="For internal use", default=False)
    parser.add_argument(
        "--test_data_file", type=str, help="Defining the path to the test file containing the answers", default=None
    )
    parser.add_argument(
        "--cache_dir", type=str, help="Cache dir for the dataset download", default=None, required=False
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Don't evaluate, just verify that the format and ids are correct.",
        default=False,
    )
    args = parser.parse_args()

    main(args)