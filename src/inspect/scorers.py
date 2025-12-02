from inspect_ai.scorer import accuracy, pattern, stderr

yesno_scorer = pattern(
    r'answer:\s*(yes|no)', 
    ignore_case=True,
)