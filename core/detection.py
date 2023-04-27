import cv2 as cv
import numpy as np
import time
import torch

from backbones.torchRetina import TorchRetina
import torch.backends.cudnn as cudnn
from components.functions import PriorBox, decode, py_cpu_nms, decode_landm

# config
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if not torch.cuda.is_available():
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
class RetinaDetector():
    def __init__(self) -> None:
        # model initialization
        torch.set_grad_enabled(False)
        self.net = TorchRetina()
        self.net = load_model(self.net, 'weights\mobilenet0.25_Final.pth')
        self.net.eval()
        cudnn.benchmark = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)

        # testing configuration
        self.resize = 1
        self.conf_thresh = .02
        self.nms_thresh = .4
        self.vis_thresh = .8
        self.p_top_k = 5000
        self.p_keep_top_k = 750

    def detect(self, frame=cv.imread('', cv.IMREAD_COLOR)):
        img = np.float32(frame)

        # processing
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        # forwarding
        t = time.time()
        loc, conf, landms = self.net(img)
        # print(f'net forward time: {time.time() - t}')
        
        pBox = PriorBox(image_size=(im_height, im_width))
        priors = pBox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.conf_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.p_top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.p_keep_top_k, :]
        landms = landms[:self.p_keep_top_k, :]

        
        # print(f'total time processing: {time.time() - t}')
        # print(dets)

        # process the returned result for recognizer
        # extract faces
        faces = []
        # extract landmarks
        ret_landms = []
        
        for idx in range(len(dets)):
            bbox = dets[idx]
            # skip unconf boxes
            if bbox[4] < self.vis_thresh:
                continue
            bbox = list(map(int, bbox))
            faces.append([bbox[0]-20, bbox[1]-20, bbox[2]+20, bbox[3]+20, bbox[4]])  # x1 y1 x2 y2 confidence

            landm = landms[idx]
            i_width = bbox[2] - bbox[0]
            i_height = bbox[3] - bbox[1]
            b_start_x = bbox[0]
            b_start_y = bbox[1]
            ret_landm = []
            # change landmark points to scale with w,h of bbox
            for land_idx in range(0, len(landm), 2):
                land_x = landm[land_idx]
                land_y = landm[land_idx + 1]

                land_w = (land_x - b_start_x) / i_width
                land_h = (land_y - b_start_y) / i_height
                ret_landm.append(land_w)
                ret_landm.append(land_h)
            ret_landms.append(ret_landm)


        # # show results
        # for idx in range(len(faces)):
        #     x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
        #     demo_face = frame[y1:y2, x1:x2]
        #     demo_face = cv.resize(demo_face, (112, 112), interpolation=cv.INTER_AREA)
        #     # landm = list(map(int, ret_landms[idx]))
        #     # landms
        #     cv.circle(demo_face, (int(ret_landms[idx][0] * 112), int(ret_landms[idx][1] * 112)), 1, (0, 0, 255), 4)
        #     cv.circle(demo_face, (int(ret_landms[idx][2] * 112), int(ret_landms[idx][3] * 112)), 1, (0, 255, 255), 4)
        #     cv.circle(demo_face, (int(ret_landms[idx][4] * 112), int(ret_landms[idx][5] * 112)), 1, (255, 0, 255), 4)
        #     cv.circle(demo_face, (int(ret_landms[idx][6] * 112), int(ret_landms[idx][7] * 112)), 1, (0, 255, 0), 4)
        #     cv.circle(demo_face, (int(ret_landms[idx][8] * 112), int(ret_landms[idx][9] * 112)), 1, (255, 0, 0), 4)

        # cv.imshow('detected', demo_face)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
                
        # for bbox in dets:
        #     # skip unconf boxes
        #     if bbox[4] < self.vis_thresh:
        #         continue
        #     conf_s = f'{bbox[4]:.4f}'
        #     bbox = list(map(int, bbox))
        #     # cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        #     # cv.putText(frame, conf_s, (bbox[0], bbox[1] + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        #     faces.append([bbox[0], bbox[1], bbox[2], bbox[3]])  # x1 y1 x2 y2
        
        return faces, ret_landms
    
class CascadeDetector():
    def __init__(self) -> None:
        self.face_cascade = cv.CascadeClassifier('weights\haarcascade_profileface.xml')
        
    def detect(self, frame=cv.imread('', cv.IMREAD_COLOR)):
        img = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.3, 4)
        return faces, []

if __name__ == '__main__':
    # detector = CascadeDetector()
    # cam = cv.VideoCapture(0)
    # while True:
    #     ret, frame = cam.read()
    #     if not ret:break
    #     bboxs = detector.detect(frame)
    #     for b in bboxs:
    #         cv.rectangle(frame, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (0, 255, 255), 1)
    #     cv.imshow('frame', frame)
    #     cv.waitKey(0)
    pass