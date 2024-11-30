from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # Load model
# result = model.predict('input/image.png', save=True)  # creates 'runs/detect/exp/image.png'
result = model.track('input/input_video.mp4', conf=0.2  ,save=True)  # creates 'runs/detect/exp/image.png'

print(result)

# print("boxes: ")

# for box in result[0].boxes :
#     print("box: ", box)