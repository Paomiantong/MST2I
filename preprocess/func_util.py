import re
from typing import Any, Dict, List

pat = re.compile(r"\[([^\]]+) : ([^\]]+?)\]")


def gen_func_call(batch: Dict[str, List[Any]]):
    func_call_list = []
    for src, utt, intent in zip(
        batch["src_annot_utt"], batch["fixed_tgt_annot_utt"], batch["intent_str"]
    ):
        params = ", ".join(f"{m[0]}={m[1]}" for m in pat.findall(utt))
        func_call_list.append(f"{intent}({params})")
    return {"func_call": func_call_list}


def gen_slots_str(batch: Dict[str, List[Any]]):
    func_call_list = []
    for utt in batch["fixed_tgt_annot_utt"]:
        params = " ".join(f"[{m[0]} : {m[1]}]" for m in pat.findall(utt))
        func_call_list.append(params)
    return {"fixed_slots_str": func_call_list}
