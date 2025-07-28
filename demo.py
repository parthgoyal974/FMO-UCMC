from ultralytics import YOLO
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np
import torch

# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.model = YOLO('pretrained/yolov5nu.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model to device: {device}")
        self.model.to(device)

    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 使用 RTDETR 进行推理  
        results = self.model(frame,imgsz = 1088)

        det_id = 0
        for box in results[0].boxes:
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id  = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets
    

def main(args):

    class_list = [2,5,7]

    cap = cv2.VideoCapture(args.video)

    # 获取视频的 fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = cv2.VideoWriter('output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # 打开一个cv的窗口，指定高度和宽度
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para)

    
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)

    # 循环读取视频帧
    detection_interval=3
    frame_id = 1
    dets = []
    while True:
        ret, frame_img = cap.read()
        if not ret:  
            print("################")
            break
        
        # dets = detector.get_dets(frame_img,args.conf_thresh,class_list)
        if frame_id % detection_interval == 1 :
        # Run YOLO detection
            dets = detector.get_dets(frame_img, args.conf_thresh, class_list)
        tracker.update(dets,frame_id)
        
        for trk in tracker.trackers:
            print(f"[DEBUG] FRAME ID={frame_id}")
            print(f"[DEBUG] Tracker ID={trk.id} status={trk.status}")

    
            x = trk.kf.x[0, 0]
            y = trk.kf.x[2, 0]
            w, h = trk.w, trk.h
            u, v = detector.mapper.xy2uv(x, y)
            print(f"[DEBUG] (u,v)=({u},{v}), box size=({w},{h})")

            if w <= 0 or h <= 0:
                print(f"[DEBUG] Skipping draw: invalid box size for ID={trk.id}")
                continue
            box_left = int(u - w / 2)
            box_top = int(v - h)
            if (box_left < 0 or box_top < 0 or 
                box_left + w > frame_img.shape[1] or box_top + h > frame_img.shape[0]):
                print(f"[DEBUG] Skipping draw: box out of image bounds for ID={trk.id}")
                continue
            cv2.rectangle(frame_img, (box_left, box_top), (box_left + int(w), box_top + int(h)), (0, 255, 0), 2)
            cv2.putText(frame_img, str(trk.id), (box_left, box_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"[DEBUG] Drew box for Tracker ID={trk.id} at ({box_left},{box_top}), size ({w},{h})")


        frame_id += 1


        # 显示当前帧
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)
    
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()



parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "demo/demo.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "demo/cam_para.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
args = parser.parse_args()

main(args)



