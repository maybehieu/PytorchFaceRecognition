import torch, os, glob, numpy as np, cv2, argparse
from core.detection import RetinaDetector
from core.recognition import ArcRecognizer
from numpy.linalg import norm
@torch.no_grad()
class Handler():
    def __init__(self, database_path) -> None:
        self.arcface = ArcRecognizer()
        self.retina = RetinaDetector()
        self.mean_face_database = []
        self.face_database = []
        self.image_size = (112, 112) # for arcface
        # True to use mean-feature verification, False for single-feature verification (take more time)
        self.verify_mode = False   
        self.database_state = False    

        # initialize database automatically
        self.init_identity_database(database_path)

    def init_identity_database(self, parent_folder_path='database'):
        # check if database needs update or not
        if self.database_state == True:
            return
        
        # reset database
        self.mean_face_database = []
        self.face_database = []

        for identity_folder in os.listdir(parent_folder_path):
            
            
            # calculating features of one identity
            feat = []
            blobs = []
            print('Reading folder: ' + identity_folder)
            for file in glob.glob(os.path.join(parent_folder_path, identity_folder) + '/*.png') + \
                glob.glob(os.path.join(parent_folder_path, identity_folder) + '/*.jpg'):
                
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                faces, landms = self.retina.detect(img)
                # fool proof for many faces detected in one image (registration will denied this)
                for idx in range(len(faces)):
                    # extract face bounding box
                    x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
                    # crop face from image with face bbox
                    face_frame = img[y1:y2, x1:x2]
                    # resize face to desired size
                    face_frame = cv2.resize(face_frame, self.image_size, interpolation=cv2.INTER_AREA)
                    # process landmark points from 'scaled' to actual 'coordinates'
                    landmk = [landm * self.image_size[0] for landm in landms[idx]]
                    # get image blob
                    blob = self.arcface.get_image(face_frame, landmk)
                    blobs.append(blob)
                    # # single image forwarding
                    # mean_feat.append(self.arcface.forward(blob))
            if len(blobs) != 0:
                # extract feature vectors
                feat = self.arcface.forward_many(blobs, len(blobs))
            if len(feat) != 0:
                # create holder vector for face feature (has 1 dimension, 1024 rows)
                mean_feat = np.zeros(shape=(1, 1024))
                # calculate 'mean/average' feature vector (sum/ n of vector on cols)
                mean_feat[0] = np.mean(feat, axis=0)
                print(f'Extracted feature for identity {identity_folder}: ', mean_feat)
                # save mean feature vector to mean face database
                self.mean_face_database.append((identity_folder, mean_feat))
                # save all feature vectors to (single) face database
                self.face_database.append((identity_folder, feat))
        print(self.mean_face_database)
        print(self.face_database)
        self.database_state = True
    
    
    """
        This function take in an image, process any face detected within and return 
        a frame with face's bounding boxes, name and confidence score
    """
    def recognize(self, img=cv2.imread('', cv2.IMREAD_COLOR)):
        # detect face from image
        faces, landms = self.retina.detect(img)
        # iterate through each image detected in frame
        for idx in range(len(faces)):
            # getting face bounding box coordinates
            x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
            # cut face from image
            face_frame = img[y1:y2, x1:x2]
            # resize image to desired size
            face_frame = cv2.resize(face_frame, (112, 112), interpolation=cv2.INTER_AREA)
            # get 'this' face landmarks
            landmk = landms[idx]
            # get input blob
            blob = self.arcface.get_image(face_frame, landmk)
            # get face feature
            feat = self.arcface.forward(blob)

            """
                NOTE:
                This step specify for checking multiple feature vector from ONE person.
                Which mean we can either check with all single-feature vector (come from single image) or
                average-feature vector (normalize from all the single-feature vector)
                USAGE:
                Change self.verify_mode to  True to verify using the mean identity face-feature,
                                            False to verify using all the identity face-feature
            """
            # find match person
            match_name = ''
            match_score = 0
            database = None
            if self.verify_mode == True:
                database = self.mean_face_database
            else:
                database = self.face_database
            for identity in database:
                # identity
                name = identity[0]
                # feature vector(s)
                match_feats = identity[1]
                # print('Calculating for: ' + name)
                for f in match_feats:
                    a = np.squeeze(feat)
                    b = np.squeeze(f)
                    cosine_similarity = np.dot(a, b) / (norm(a)*norm(b))
                    # find best match
                    if cosine_similarity > match_score:
                        match_name = name
                        match_score = cosine_similarity
                # print()
            
            # draw processed result
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(img, 'Name: ' + match_name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
            cv2.putText(img, 'Confidence: ' + str(match_score), (x1, y1 - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
        return img
        

    def cal_and_app_feature(self, features, identity_name):
        if len(features) == 0:
            return
        for f in feature:
            self.face_database.append((identity_name, f))
        mean_f = np.zeros((1, 1024))
        mean_f = np.mean(features, axis=0)
        self.mean_face_database.append((identity_name, mean_f))

    def register_identity(self, img=cv2.imread('', cv2.IMREAD_COLOR), identity=''):
        # detect face from image
        faces, landms = self.retina.detect(img)
        # initialize parameters
        msg = ''
        # check if only one face detected in frame
        if len(faces) > 1:
            msg = 'More than one face detected!'
            print(msg)
            # print(type([]), type(img))
            return [], img
        else:
            # this for loop is only fool-proof, program logic will only add ONE person in ONE frame at a time.
            for idx in range(len(faces)):
                # getting face bounding box coordinates
                x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
                # cut face from image
                face_frame = img[y1:y2, x1:x2]
                # resize image to desired size
                face_frame = cv2.resize(face_frame, (112, 112), interpolation=cv2.INTER_AREA)
                # get 'this' face landmarks
                landmk = landms[idx]
                # get input blob
                blob = self.arcface.get_image(face_frame, landmk)
                # get face feature
                feat = self.arcface.forward(blob)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(img, str(faces[idx][4]), (x1, y1 + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
                # print(type(feat), type(img))
                self.database_state = False
                return feat, img
        return [], img
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial Recognition Application')
    parser.add_argument('--mode', '-m', type=int, default=0, help='0 - register new face into system, 1 - face recognition')
    parser.add_argument('--imgs', '-i', type=str, default='', help='if empty, auto search for webcam, else - a path to a folder contains images of new face')
    parser.add_argument('--database', '-dp', type=str, default='database', help='Path to parent \'database\' path, contain sub-folders in which contain face images')
    args = vars(parser.parse_args())
    
    # # initialize main handler
    handler = Handler(args['database'])

    # # initialize program
    STATE = True
    FACE_NUM_THRESH = 20
    while STATE:
        cam = cv2.VideoCapture(0)
        # face registering
        if args['mode'] == 0:
            # initialaize parameters
            # identity_folder_path = input('Identity folder path(with forward "/" slash): ')
            database_path = args['database']
            identity_name = input('Input identity\'s name: ')
            identity_path = os.path.normpath(os.path.join(database_path + '/' + identity_name))
            # check if identity folder exist, create if not
            if not os.path.exists(identity_path):
                os.mkdir(identity_path)
            # get current number of images belong to identity (if exist)
            face_count = len(glob.glob(os.path.join(identity_path, '/*.png')) + \
            glob.glob(os.path.join(identity_path, '/*.jpg')))
            register_count = 0
            features = []
            # loop
            while True:
                ret, frame = cam.read()
                original = frame.copy()
                if not ret:
                    handler.cal_and_app_feature(features, identity_name)
                    break
                if register_count > 20 or face_count > 30:
                    handler.cal_and_app_feature(features, identity_name)
                    print('Faces belong to identity exceeded require threshold, delete old images or continue with normal functionality!')
                    break
                feature, ret_frame = handler.register_identity(frame)
                # show result if process registering success
                if len(feature) != 0:
                    # append to current feature vector (of one identity)
                    features.append(feature)
                    # save image for later database initialization
                    cv2.imwrite(os.path.normpath(os.path.join(database_path + '/' + identity_name, identity_name + '_' + str(face_count) + '.png')),
                                original)
                    face_count += 1
                print('Press any key to continue registering!')
                cv2.imshow('frame', ret_frame)
                cv2.waitKey(0)
            cv2.destroyAllWindows()
            args['mode'] = 1
        # face recognition
        elif args['mode'] == 1:
            # re-initialize face database in case of new registration
            handler.init_identity_database(args['database'])
            # loop
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                ret_frame = handler.recognize(frame)
                # show result if process registering success
                cv2.imshow('Recognition', ret_frame)
                if cv2.waitKey(5) == ord(str('q')):
                    break
            cv2.destroyAllWindows()
            STATE = False