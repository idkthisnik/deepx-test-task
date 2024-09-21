import random
from typing import Dict, List, Union


async def get_boxes_list(model_results: list) -> List[Union[str, float, list]]:
    result = []
    boxes = model_results[0].boxes.xyxy.tolist()
    classes = model_results[0].boxes.cls.tolist()
    names = model_results[0].names
    confidences = model_results[0].boxes.conf.tolist()
    
    for box, cls, conf in zip(boxes, classes, confidences):
        prediction = {
            "category": names[int(cls)],
            "confidence": conf,
            "bounding_box": box
        }
        result.append(prediction)
    return result


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
