from ultralytics import YOLO

model = YOLO('models/weights/yolov5_last.pt')  # Load model
# result = model.predict('input/image.png', save=True)  # creates 'runs/detect/exp/image.png'
result = model.predict('input/input_video.mp4', conf=0.2  ,save=True)  # creates 'runs/detect/exp/image.png'

print(result)

print("boxes: ")

for box in result[0].boxes :
    print("box: ", box)