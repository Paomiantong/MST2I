import os
from pathlib import Path
from string import Template
from glob import glob
from typing import Dict, Tuple

cwd = os.path.abspath(os.path.dirname(__file__))

PROMPTS: Dict[str, Tuple[Template, Template]] = {}

for md in glob(cwd + "/*.md"):
    with open(md, "r") as f:
        name = Path(md).stem
        prompt_tpl, answer_tpl = f.read().split("<SEP>")
        PROMPTS[name] = (Template(prompt_tpl.lstrip()), Template(answer_tpl.lstrip()))


def get_prompt(name):
    assert name in PROMPTS
    return PROMPTS[name]
