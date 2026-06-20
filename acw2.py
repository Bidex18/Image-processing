#!/usr/bin/env python3

import os
import argparse
from ultralytics import YOLO
import cv2


# Mapping from class index to (sign_number, sign_name)
CLASS_MAP = {

  0:  'Roundabout',
  1:  'Traffic Lights', 
  2:  'No entry' ,
  3:  'One Way',
 17:  '30MPH',
  5:  'National Speed Limit',
  6:  'Double End Road',
  7:  'Warning',
  8:  'Roadworks',
  9:  'No Turn',
  10:  '40MPH',
  11:  'Double Way',
  12:  'Two Way',
  13:  'Bending',
  14:  'Give Way',
  15:  '50MPH',
  16:  '20MPH',
  4:  'Pedestrians',
  18:  'Stop',
  19:  'Duck Crossing',
  20:  'Notice Board',
  21:  'Double bended',
  22:  'Mini Roundabout',
  23:  'No right turn',
  24:  'No left turn',  
  24:  'No Stopping',
  25:  'No waiting',
  26:  'No overtaking',
  27:  'Ahead Only',
  28:  'Children crossing',
  29:  'Patrol',
  30:  'Narrow Road',
  31:  'Sharp Left',
  32:  'Sharp Right',
  33:  'slippery road',
  34:  'Zebra Crossing',
}
  
def parse_args():
    parser = argparse.ArgumentParser(description='Detect road signs in a single image')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--output', '-o', default=None, help='Path to output text file')
    parser.add_argument('--interactive', action='store_true', help='Display results in a window')
    return parser.parse_args()
def name(name):
    return f"'{name}'" if " " in name else name

def main():
    args = parse_args()
    img_path = args.image
    out_path = args.output or "output.txt"

    # load your trained model 
    model = YOLO('weights.pt')

    # run inference
    results = model.predict(source=img_path, conf=0.25, save=False)[0]

    # prepare output lines
    with open(out_path, "w", encoding="utf8") as f:
        # for each detection box
        h, w = results.orig_shape[:2]
        for box, cls_idx, conf in zip(results.boxes.xyxy,
                                      results.boxes.cls,
                                      results.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            # normalized center + size
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            sign_num = int(cls_idx.item())
            sign_name = CLASS_MAP.get(sign_num, f"class_{sign_num}")
            conf_val = conf.item()

            fn = os.path.basename(img_path)
            fn = name(fn)

            # write one line: 10 comma-separated fields
            line = ",".join([
                fn,
                str(sign_num),
                sign_name,
                f"{xc:.6f}",
                f"{yc:.6f}",
                f"{bw:.6f}",
                f"{bh:.6f}",
                "0",              # frame_number
                "0",              # timestamp
                f"{conf_val:.6f}"
            ])
            f.write(line + "\n")

    print(f"Wrote {len(results.boxes):d} detections to {out_path}")
            

    # Interactive display
    if args.interactive:
        img = cv2.imread(img_path)
        for box, cls_idx, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{CLASS_MAP[int(cls_idx.item())]} {conf.item():.2f}"
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Detections", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
