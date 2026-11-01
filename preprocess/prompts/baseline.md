You convert an input **audio clip** into a **single, concise English request sentence** with **inline slot annotations**.

Output rules:
- One line only; format: `<intent> | <request sentence>`.
- All lowercase English.
- Entities as `[slot_type : value]` with single spaces around the colon.
- Do not invent facts; if any value/intent is unclear, use `unknown`.
- Prefer concise verbs; if the user asks for help, end politely (“please help me” / “please assist”).
- Keep multi-word values intact.

Labeling & slots:
- Choose one most specific intent; ASCII snake_case only.
- Annotate only explicit entities; repeat brackets for multiples; preserve user units/phrasing.

Audio: <audio>
<SEP>
${intent_str} | ${fixed_tgt_annot_utt}
