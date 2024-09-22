import csv

import requests
import typer
from PIL import Image, ImageDraw, ImageFont

import utils


def send_image(source: str = typer.Option(..., help="Path to the image")) -> dict:
    api_url = "http://0.0.0.0:8000/predict/"
    with open(source, "rb") as f:
        files = {"file": f}
        response = requests.post(url=api_url, files=files)
        response_data = response.json()
        save_to_csv(response_data)
        save_to_jpg(f, response_data)


def save_to_csv(response_data: dict) -> None:
    predictions = response_data["predictions"]
    
    with open("result/output.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["box", "category"])
        for prediction_object in predictions:
            writer.writerow([prediction_object["bounding_box"], prediction_object["category"]])


def save_to_jpg(image_path: str, response_data: dict) -> None:
    predictions = response_data["predictions"]
    colors = utils.get_random_colors_for_categories(predictions)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    fnt = ImageFont.truetype(font="Arial Bold" , size=30)
    
    for prediction_object in predictions:
        category_color = colors[prediction_object["category"]]
        draw.rectangle(prediction_object["bounding_box"], outline=category_color, width=5)
        bounding_box_text = utils.get_bounding_box_text(prediction_object)
        bounding_box_cords = utils.get_bounding_box_cords(prediction_object)
        draw.text(xy=bounding_box_cords, text=bounding_box_text, font=fnt, fill=category_color)
            
    image.save("result/output.jpg")


if __name__ == "__main__":
    typer.run(send_image)
