"""Recursively preserve native-English records and user-authored message content."""
def english(value):
    if isinstance(value, list):
        return [english(item) for item in value]
    if isinstance(value, dict):
        return {key: english(item) for key, item in value.items()}
    return value
