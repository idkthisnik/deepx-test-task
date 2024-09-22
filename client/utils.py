import random
from typing import Dict


def get_bounding_box_text(prediction_object: dict) -> str:
    formatted_confidence = "{:.2f}".format(prediction_object["confidence"])
    category_name = prediction_object["category"]
    return f"{category_name} {formatted_confidence}"
    
    
def get_bounding_box_cords(prediction_objects: dict) -> tuple:
    x1, y1, _, _ = prediction_objects["bounding_box"]
    return (x1, y1 - 30)


def get_random_colors_for_categories(predicionts) -> Dict[str, str]:
    categories = set(p["category"] for p in predicionts)
    result = {}
    for c in categories:
        levels = range(32,256,32)
        result[c] = tuple(random.choice(levels) for _ in range(3))
    return result
