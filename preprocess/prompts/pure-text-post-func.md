You convert an input **text utterance** into a **single, concise English request sentence** with **inline slot annotations**.

Steps: (1) read the input text in ${src_locale} faithfully (no normalization), (2) translate the request to natural English, (3) compress into one short user-intent sentence, (4) infer exactly one lowercase snake_case intent (prefer domain_action; if unclear, use "unknown"), (5) convert that intent and any explicit entities into a function call, (6) annotate explicit entities inline.

Output rules:
- Final sentence format: `<intent> | <request sentence>`.
- All lowercase English.
- Entities as `[slot_type : value]` with single spaces around the colon.
- Do not invent facts; if any value/intent is unclear, use `unknown`.
- Prefer concise verbs; if the user asks for help, end politely (“please help me” / “please assist”).
- Keep multi-word values intact.

Labeling & slots:
- Choose one most specific intent; ASCII snake_case only.
- Annotate only explicit entities; repeat brackets for multiples; preserve user units/phrasing.

Text: ${src_utt}
<SEP>
The translation from ${src_locale} to English request sentence is:
${fixed_tgt_utt}

The function call is:
${func_call}

The final translation from ${src_locale} to English request sentence is:
${intent_str} | ${fixed_tgt_annot_utt}
