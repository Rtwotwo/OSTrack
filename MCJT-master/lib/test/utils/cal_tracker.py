from lib.test.utils.tracker import KalmanBoxTracker

def getInitTracker(box):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    return KalmanBoxTracker(box)

def update_xywh(tracker, box, conf):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    tracker.update(box + conf)

def update_xyxy(tracker, box, conf):
    tracker.update(box + conf)

def xywh2xyxy(box):    # xyxy -> xywh
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]

def update(q):
    size = len(q)
    ini_box = q.popleft()
    ini_box = xywh2xyxy(ini_box)
    size -= 1
    kalman = KalmanBoxTracker(ini_box)
    for i in range(size):
        box = q.popleft()
        box = xywh2xyxy(box)
        kalman.predict()
        kalman.update(box)
    return kalman.predict().squeeze()

class KalmanTracker:
    def __init__(self, box):
        self.kalman = getInitTracker(box)