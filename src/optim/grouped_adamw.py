from typing import Dict, List, Any
import torch
from src.optim.merge_optimizer import MergedOptimizer


from typing import Dict, List, Any
import torch
from src.optim.merge_optimizer import MergedOptimizer


def grouped_adamw(
    lr: float,
    weight_decay: float,
    params: List[Dict[str, Any]],
):
    from collections import defaultdict

    groups = defaultdict(list)

    for group in params:
        group_wd = group.get("weight_decay", weight_decay)
        names = group.get("names", [])
        tensors = group["params"]

        for name, p in zip(names, tensors):

            is_sparse = "unit_emb" in name or "session_emb" in name
            is_input_ca = "input_ca" in name
            group_lr = lr * 1.0 if is_input_ca else lr

            key = (is_sparse, group_wd, group_lr)
            groups[key].append(p)

    param_groups = []

    for (is_sparse, wd, group_lr), tensors in groups.items():

        base = "sparse" if is_sparse else "dense"
        lr_tag = "input_ca" if group_lr != lr else "base"
        wd_tag = "decay" if wd > 0 else "no_decay"

        name = f"{base}_{lr_tag}_{wd_tag}"

        param_groups.append(
            {
                "params": tensors,
                "weight_decay": wd,
                "lr": group_lr,
                "name": name,
            }
        )

    return MergedOptimizer([torch.optim.AdamW(param_groups)])
