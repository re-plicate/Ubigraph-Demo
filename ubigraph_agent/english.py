"""English presentation for system records; historical conversation content stays intact."""
import re
import json
from pathlib import Path
_TABLE = json.loads(Path(__file__).with_name("english.json").read_text())
_PATTERN = re.compile(r"[\u3400-\u9fff][\u3400-\u9fff，。！？；：“”‘’、（）《》…—·\s]*")
def english(value):
    if isinstance(value, str):
        if value in _TABLE: return _TABLE[value]
        def replace(match):
            key = match.group().rstrip()
            translated = _TABLE.get(key, key)
            before = value[match.start()-1:match.start()] if match.start() else ""
            after = value[match.end():match.end()+1]
            if before and before.isascii() and before.isalnum() and translated[:1].isalnum(): translated = " " + translated
            if after and after.isascii() and after.isalnum() and translated[-1:].isalnum(): translated += " "
            return translated + match.group()[len(key):]
        return _PATTERN.sub(replace, value).translate(str.maketrans({"，": ", ", "；": "; ", "：": ": ", "。": ". ", "、": ", ", "（": " (", "）": ") "}))
    if isinstance(value, list): return [english(v) for v in value]
    if isinstance(value, dict):
        preserve = value.get("role") in ("user", "assistant") or value.get("type") == "text_delta"
        return {k: (v if preserve and k in ("content", "text") else english(v)) for k,v in value.items()}
    return value
