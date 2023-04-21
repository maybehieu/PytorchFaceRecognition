import argparse
from scipy import interpolate
from skimage import transform as trans

from backbones.torchArcface import ResNet50

import torch, torchvision, os, glob, numpy as np, cv2
@torch.no_grad()
class ArcRecognizer():
    def __init__(self, weight_path=r'weights\arcface_resnet50.pth') -> None:
        # self.model = get_model('r50', dropout=0, fp16=True).cuda()
        self.model = ResNet50(dropout=0, FP16=True).cuda()
        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()

        self.align = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.align[:, 0] += 8.0
        
        self.image_size = (112, 112)
        self.batch_size = 20


    def get_image(self, rimg=cv2.imread('', cv2.IMREAD_COLOR), landmark=[]):
        img = rimg
        # # get landmark points
        # landmark5 = np.zeros((5, 2), dtype=np.float32)
        # landmark5[0] = [landmark[0], landmark[1]]
        # landmark5[1] = [landmark[2], landmark[3]]
        # landmark5[2] = [landmark[4], landmark[5]]
        # landmark5[3] = [landmark[6], landmark[7]]
        # landmark5[4] = [landmark[8], landmark[9]]
    
        # # # aligning face
        # tform = trans.SimilarityTransform()
        # tform.estimate(landmark5, self.align)
        # M = tform.params[0:2, :]
        # img = cv2.warpAffine(rimg,
        #                      M, (self.image_size[1], self.image_size[0]),
        #                      borderValue=0.0)
        # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # #
        img_flip = np.fliplr(img)
        # print(img.shape[:2], img_flip.shape[:2], sep='\n')
        # cv2.imshow('ori', rimg)
        # cv2.imshow('fimg', img_flip)
        # cv2.imshow('rimg', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    def forward(self, input=np.zeros((2, 3, 112, 112), dtype=np.uint8)):
        imgs = torch.Tensor(input).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([1, 2 * feat.shape[1]])
        feat = feat.cpu().detach().numpy()
        return feat
    
    def forward_many(self, blobs, batch_size):
        # process images array to batch_data (for pytorch)
        batch_data = np.empty((2 * batch_size, 3, 112, 112))
        for idx in range(batch_size):
            # blob index increment by 1, batch index increment by {blob index} * 2
            # because one blob contain two images
            batch_data[2 * idx][:] = blobs[idx][0]
            batch_data[2 * idx + 1][:] = blobs[idx][1]

        # forward
        imgs = torch.Tensor(batch_data).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([batch_size, 2 * feat.shape[1]])
        feat = feat.cpu().detach().numpy()
        return feat

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    # parser.add_argument('--network', type=str, default='r50', help='backbone network')
    # parser.add_argument('--weight', type=str, default='')
    # parser.add_argument('--img', type=str, default=None)
    # args = parser.parse_args()
    # inference(args.weight, args.network, args.img)\
    pass

