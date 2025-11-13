@echo off
echo START TRAINING...

REM === PART 1 ===
python train.py --img 416 --batch 2 --epochs 30 ^
 --data D:/2025/FA25/AIL/yolo/part1.yaml ^
 --weights yolov5n.pt --device 0 ^
 --project runs/train --name p1 --exist-ok --workers 0
set W1=runs\train\p1\weights\best.pt

REM === PART 2 ===
python train.py --img 416 --batch 2 --epochs 30 ^
 --data D:/2025/FA25/AIL/yolo/part2.yaml ^
 --weights %W1% --device 0 ^
 --project runs/train --name p2 --exist-ok --workers 0
set W2=runs\train\p2\weights\best.pt

REM === PART 3 ===
python train.py --img 416 --batch 2 --epochs 30 ^
 --data D:/2025/FA25/AIL/yolo/part3.yaml ^
 --weights %W2% --device 0 ^
 --project runs/train --name p3 --exist-ok --workers 0
set W3=runs\train\p3\weights\best.pt

REM === PART 4 ===
python train.py --img 416 --batch 2 --epochs 30 ^
 --data D:/2025/FA25/AIL/yolo/part4.yaml ^
 --weights %W3% --device 0 ^
 --project runs/train --name p4 --exist-ok --workers 0

echo ------------------------------------------------
echo ✅ DONE — FINAL MODEL HERE:
echo runs/train/p4/weights/best.pt
echo ------------------------------------------------
