""" SCROLLS benchmark metric. """

from collections import defaultdict
from copy import deepcopy
import datasets

# fmt: off
from .rouge import compute_rouge, postprocess_text as rouge_postprocess_text  # From: https://huggingface.co/datasets/tau/scrolls/raw/main/metrics/rouge.py
from .exact_match import compute_exact_match  # From: https://huggingface.co/datasets/tau/scrolls/raw/main/metrics/exact_match.py
from .f1 import compute_f1  # From: https://huggingface.co/datasets/tau/scrolls/raw/main/metrics/f1.py
# fmt: on

_CITATION = """\
@misc{shaham2022scrolls,
      title={SCROLLS: Standardized CompaRison Over Long Language Sequences}, 
      author={Uri Shaham and Elad Segal and Maor Ivgi and Avia Efrat and Ori Yoran and Adi Haviv and Ankit Gupta and Wenhan Xiong and Mor Geva and Jonathan Berant and Omer Levy},
      year={2022},
      eprint={2201.03533},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
SCROLLS: Standardized CompaRison Over Long Language Sequences.
A suite of natural language datasets that require reasoning over long texts.
https://scrolls-benchmark.com/
"""

_KWARGS_DESCRIPTION = """
Compute Scrolls evaluation metric associated to each Scrolls dataset.
Args:
    predictions: list of predictions to score.
        Each prediction should be a string.
    references: list of lists of references for each example.
        Each reference should be a string.
Returns: depending on the Scrolls subset, one or several of:
    "exact_match": Exact Match score
    "f1": F1 score
    "rouge": ROUGE score

Use the following code to download the metric:
```
import os, shutil
from huggingface_hub import hf_hub_download
def download_metric():
    scrolls_metric_path = hf_hub_download(repo_id="datasets/tau/scrolls", filename="metrics/scrolls.py")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path

scrolls_metric_path = download_metric()
```

Examples:
    predictions = ["exact match example", "hello there", "general kenobi"]  # List[str]
    references = [["exact match example"], ["hello", "hi there"], ["commander kenobi"]]  # List[List[str]]

    >>> scrolls_metric = datasets.load_metric(scrolls_metric_path, 'gov_report')  # 'gov_report' or any of ["qmsum", "summ_screen_fd"]
    >>> results = scrolls_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'rouge/rouge1': 72.2222, 'rouge/rouge2': 33.3333, 'rouge/rougeL': 72.2222, 'rouge/rougeLsum': 72.2222, 'rouge/geometric_mean': 55.8136, 
    'num_predicted': 3, 'mean_prediction_length_characters': 14.6667, 'scrolls_score': 55.8136, 
    'display_keys': ['rouge/rouge1', 'rouge/rouge2', 'rouge/rougeL'], 'display': [72.2222, 33.3333, 72.2222]}

    >>> scrolls_metric = datasets.load_metric(scrolls_metric_path, 'contract_nli')  # 'contract_nli' or "quality"
    >>> results = scrolls_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact_match': 33.3333, 'num_predicted': 3, 'mean_prediction_length_characters': 14.6667, 'scrolls_score': 33.3333, 
    'display_keys': ['exact_match'], 'display': [33.3333]}

    >>> scrolls_metric = datasets.load_metric(scrolls_metric_path, 'narrative_qa')  # 'narrative_qa' or "qasper"
    >>> results = scrolls_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'f1': 72.2222, 'num_predicted': 3, 'mean_prediction_length_characters': 14.6667, 'scrolls_score': 72.2222, 
    'display_keys': ['f1'], 'display': [72.2222]}
"""

DATASET_TO_METRICS = {
    "contract_nli": {
        "metrics_to_compute": ["exact_match"],
        "scrolls_score_key": "exact_match",
        "display_keys": ["exact_match"],
    },
    "gov_report": {
        "metrics_to_compute": ["rouge"],
        "scrolls_score_key": "rouge/geometric_mean",
        "display_keys": ["rouge/rouge1", "rouge/rouge2", "rouge/rougeL"],
    },
    "narrative_qa": {
        "metrics_to_compute": ["f1"],
        "scrolls_score_key": "f1",
        "display_keys": ["f1"],
    },
    "qasper": {
        "metrics_to_compute": ["f1"],
        "scrolls_score_key": "f1",
        "display_keys": ["f1"],
    },
    "qmsum": {
        "metrics_to_compute": ["rouge"],
        "scrolls_score_key": "rouge/geometric_mean",
        "display_keys": ["rouge/rouge1", "rouge/rouge2", "rouge/rougeL"],
    },
    "summ_screen_fd": {
        "metrics_to_compute": ["rouge"],
        "scrolls_score_key": "rouge/geometric_mean",
        "display_keys": ["rouge/rouge1", "rouge/rouge2", "rouge/rougeL"],
    },
    "quality": {
        "metrics_to_compute": ["exact_match"],
        "scrolls_score_key": "exact_match",
        "display_keys": ["exact_match"],
    },
    "quality_hard": {
        "metrics_to_compute": ["exact_match"],
        "scrolls_score_key": None,
        "display_keys": ["exact_match"],
    },
}


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Scrolls(datasets.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._compute_helper_kwargs_fn = {
            "rouge": lambda: {
                "metric_fn": compute_rouge,
                "agg_fn": max,
                "metric_fn_kwargs": {"use_stemmer": False},
                "metric_returns_per_example": True,
                "transform_single_input_fn": lambda text: rouge_postprocess_text(text),
                "transform_result_fn": lambda output: {
                    key: (value[0] if isinstance(value, list) else value).fmeasure * 100
                    for key, value in output.items()
                },
                "transform_aggregated_result_fn": lambda output: output.update(
                    {"geometric_mean": (output["rouge1"] * output["rouge2"] * output["rougeL"]) ** (1.0 / 3.0)}
                )
                or output,
            },
            "exact_match": lambda: {
                "metric_fn": compute_exact_match,
                "agg_fn": None,  # compute_exact_match already takes max
                "transform_result_fn": lambda output: {None: output},
            },
            "f1": lambda: {
                "metric_fn": compute_f1,
                "agg_fn": None,  # compute_f1 already takes max
                "transform_result_fn": lambda output: {None: output},
            },
        }

        custom_metrics = (
            [metric for metric in self.config_name.split(",") if len(metric) > 0]
            if self.config_name.startswith(",")
            else None
        )
        if custom_metrics is not None:
            for metric in custom_metrics:
                if metric not in self._compute_helper_kwargs_fn:
                    raise KeyError(
                        f"You should supply a metric name selected in {list(self._compute_helper_kwargs_fn.keys())}"
                    )
            self._metrics_to_compute = custom_metrics
        else:
            if self.config_name not in DATASET_TO_METRICS:
                raise KeyError(f"You should supply a configuration name selected in {list(DATASET_TO_METRICS.keys())}")
            self._metrics_to_compute = DATASET_TO_METRICS[self.config_name]["metrics_to_compute"]

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Sequence(datasets.Value("string")),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def convert_from_map_format(self, id_to_pred, id_to_labels):
        index_to_id = list(id_to_pred.keys())
        predictions = [id_to_pred[id_] for id_ in index_to_id]
        references = [id_to_labels[id_] for id_ in index_to_id]
        return {"predictions": predictions, "references": references}

    def _compute(self, predictions, references):
        metrics = {}
        for metric in self._metrics_to_compute:
            result = _compute_helper(
                deepcopy(predictions),
                deepcopy(references),
                **self._compute_helper_kwargs_fn[metric](),
            )
            metrics.update(
                {(f"{metric}/{key}" if key is not None else metric): value for key, value in result.items()}
            )
        metrics["num_predicted"] = len(predictions)
        prediction_lengths = [len(prediction) for prediction in predictions]
        metrics["mean_prediction_length_characters"] = sum(prediction_lengths) / len(prediction_lengths)

        metrics = {key: round(value, 4) for key, value in metrics.items()}

        if self.config_name in DATASET_TO_METRICS:
            scrolls_score_key = DATASET_TO_METRICS[self.config_name]["scrolls_score_key"]
            if scrolls_score_key is not None:
                metrics["scrolls_score"] = metrics[scrolls_score_key]
            else:
                metrics["scrolls_score"] = None

            display_keys = DATASET_TO_METRICS[self.config_name]["display_keys"]
            metrics["display_keys"] = display_keys
            metrics["display"] = []
            for display_key in display_keys:
                metrics["display"].append(metrics[display_key])

        return metrics


def _compute_helper(
    predictions,
    references,
    metric_fn,
    agg_fn,
    metric_fn_kwargs=None,
    transform_single_input_fn=None,
    transform_result_fn=None,
    transform_aggregated_result_fn=None,
    metric_returns_per_example=False,
):
    if metric_fn_kwargs is None:
        metric_fn_kwargs = {}

    if agg_fn is None:
        assert metric_returns_per_example is False

    if transform_single_input_fn is not None:
        predictions = [transform_single_input_fn(prediction) for prediction in predictions]
        references = [
            [transform_single_input_fn(reference) for reference in reference_list] for reference_list in references
        ]

    if transform_result_fn is None:
        transform_result_fn = lambda x: x
        do_transform_result = False
    else:
        do_transform_result = True

    if transform_aggregated_result_fn is None:
        transform_aggregated_result_fn = lambda x: x

    if agg_fn is not None:
        # Required when the metric doesn't do the aggregation we need
        scores = defaultdict(list)
        if metric_returns_per_example is False:
            # If when given a list of prediction and references the metric returns an aggregated score,
            # we need to compute the metric for each prediction and reference and then aggregate the results.
            # This is only an issue when we want to get the best aggregated score (e.g. max) for prediction
            # with multiple references.
            for prediction, reference_list in zip(predictions, references):
                prediction_scores = defaultdict(list)
                for reference in reference_list:
                    result = transform_result_fn(metric_fn([prediction], [reference], **metric_fn_kwargs))
                    for key in result:
                        prediction_scores[key].append(result[key])
                for key in prediction_scores:
                    scores[key].append(agg_fn(prediction_scores[key]))
        else:
            # Flatten the references and then aggregate per prediction with agg_fn
            mapping = [[] for _ in range(len(predictions))]
            flattened_predictions = []
            flattened_references = []
            for i, prediction in enumerate(predictions):
                for reference in references[i]:
                    flattened_predictions.append(prediction)
                    flattened_references.append(reference)
                    mapping[i].append(len(flattened_references) - 1)

            results = metric_fn(flattened_predictions, flattened_references, **metric_fn_kwargs)
            if isinstance(results, dict):
                # Convert a dictionary with lists per key to a list with dictionary with the same keys per element
                results_list = [{k: None for k in results} for _ in range(len(flattened_predictions))]
                for k, v in results.items():
                    for i in range(len(v)):
                        results_list[i][k] = v[i]
            else:
                results_list = results

            if do_transform_result:
                for i in range(len(results_list)):
                    results_list[i] = transform_result_fn(results_list[i])

            for reference_indexes in mapping:
                prediction_scores = defaultdict(list)
                for reference_index in reference_indexes:
                    result = results_list[reference_index]
                    for key in result:
                        prediction_scores[key].append(result[key])
                for key in prediction_scores:
                    scores[key].append(agg_fn(prediction_scores[key]))

        return transform_aggregated_result_fn({key: sum(value) / len(value) for key, value in scores.items()})
    else:
        return transform_aggregated_result_fn(
            transform_result_fn(metric_fn(predictions, references, **metric_fn_kwargs))
        )

