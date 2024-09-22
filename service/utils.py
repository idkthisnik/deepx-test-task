from typing import List, Union


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
