from inspect_ai.scorer import pattern

yesno_scorer = pattern(
    r'answer:\s*(yes|no)', 
    ignore_case=True,
)