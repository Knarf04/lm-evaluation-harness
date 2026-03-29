"""Generate LongBench-E context-length bucket configs.

Creates task YAMLs, category group YAMLs, and top-level group YAMLs
for the three LongBench-E length buckets: 0-4k, 4-8k, 8k+.

Usage:
    python _generate_bucket_configs.py
"""

import os

BUCKETS = [
    ("0_4k", "filter_0_4k", "0-4k"),
    ("4_8k", "filter_4_8k", "4-8k"),
    ("8k_plus", "filter_8k_plus", "8k+"),
]

# Category -> list of dataset names (without _e suffix)
CATEGORIES = {
    "single": ["qasper", "multifieldqa_en"],
    "multi": ["hotpotqa", "2wikimqa"],
    "summarization": ["gov_report", "multi_news"],
    "fewshot": ["trec", "triviaqa", "samsum"],
    "code": ["lcc", "repobench-p"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
}

CATEGORY_ALIASES = {
    "single": "Single-Document QA",
    "multi": "Multi-Document QA",
    "summarization": "Summarization",
    "fewshot": "Few-shot Learning",
    "code": "Code Completion",
    "synthetic": "Synthetic Tasks",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_task_yamls():
    """Generate per-bucket task YAMLs (39 files)."""
    for bucket_suffix, filter_fn, _ in BUCKETS:
        for category, datasets in CATEGORIES.items():
            for ds in datasets:
                filename = f"{ds}_e_{bucket_suffix}.yaml"
                content = (
                    f"include: {ds}_e.yaml\n"
                    f"task: longbench_{ds}_e_{bucket_suffix}\n"
                    f"process_docs: !function metrics.{filter_fn}\n"
                    f"tag:\n"
                    f"  - longbench_{category}_tasks_e_{bucket_suffix}\n"
                    f"  - longbench_tasks_e_{bucket_suffix}\n"
                )
                filepath = os.path.join(SCRIPT_DIR, filename)
                with open(filepath, "w") as f:
                    f.write(content)
                print(f"  Created {filename}")


def generate_category_group_yamls():
    """Generate per-bucket category group YAMLs (18 files)."""
    for bucket_suffix, _, bucket_label in BUCKETS:
        for category, datasets in CATEGORIES.items():
            alias = CATEGORY_ALIASES[category]
            filename = f"_longbench_{category}_e_{bucket_suffix}.yaml"
            tasks = "\n".join(
                f"  - longbench_{ds}_e_{bucket_suffix}" for ds in datasets
            )
            content = (
                f"group: longbench_{category}_e_{bucket_suffix}\n"
                f'group_alias: "{alias} (LongBench-E {bucket_label})"\n'
                f"task:\n"
                f"{tasks}\n"
                f"aggregate_metric_list:\n"
                f"  - metric: score\n"
                f"    weight_by_size: False\n"
                f"metadata:\n"
                f"  version: 0.0\n"
            )
            filepath = os.path.join(SCRIPT_DIR, filename)
            with open(filepath, "w") as f:
                f.write(content)
            print(f"  Created {filename}")


def generate_top_level_group_yamls():
    """Generate per-bucket top-level group YAMLs (3 files)."""
    for bucket_suffix, _, _ in BUCKETS:
        filename = f"_longbench_e_{bucket_suffix}.yaml"
        category_groups = "\n".join(
            f"  - longbench_{cat}_e_{bucket_suffix}" for cat in CATEGORIES
        )
        content = (
            f"group: longbench_e_{bucket_suffix}\n"
            f"task:\n"
            f"{category_groups}\n"
            f"aggregate_metric_list:\n"
            f"  - metric: score\n"
            f"    weight_by_size: False\n"
            f"metadata:\n"
            f"  version: 0.0\n"
        )
        filepath = os.path.join(SCRIPT_DIR, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  Created {filename}")


if __name__ == "__main__":
    print("Generating task YAMLs (39 files)...")
    generate_task_yamls()
    print("\nGenerating category group YAMLs (18 files)...")
    generate_category_group_yamls()
    print("\nGenerating top-level group YAMLs (3 files)...")
    generate_top_level_group_yamls()
    print("\nDone! Generated 60 YAML files.")
