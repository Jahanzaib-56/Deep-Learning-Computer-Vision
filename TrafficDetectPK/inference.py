from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict(
    source = 'Sargodha-Traffic.mp4',
    save = True,
    project = 'runs',
    name = 'inference-output',
    conf = 0.4,
    iou = 0.5,
    show = True,
)

for r in results:
    pass

print("Inference Complete.....")