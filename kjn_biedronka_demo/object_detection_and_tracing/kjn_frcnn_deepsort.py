import os
import time
from distutils.util import strtobool
import cv2
from object_detection_and_tracing.deep_sort.deep_sort import DeepSort
from object_detection_and_tracing.kjn_object_detection import KjnFastRCNN
# from util import draw_bboxes
import ffmpeg
import pathlib

def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    return rotateCode

def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)

def load_images(path):
    images = []
    valid_images = ['.png', '.jpg', '.jpeg']
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images

def load_movies(path):
    images = []
    valid_movies = [".mp4", ".avi"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_movies:
            continue
        images.append(os.path.join(path, f))
    return images

class KjnDetectorCamera(object):
    def __init__(self, video_in_path, video_out_path, display=False):
        super().__init__()
        self.video_in_path = str(video_in_path)
        self.video_out_path = str(video_out_path)
        self.display = display
        use_cuda = True
        if self.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", 800, 600)
        self.vdo = cv2.VideoCapture()
        self.frcnn= KjnFastRCNN()
        self.deepsort = DeepSort("deep_sort/deep/checkpoint/ckpt.t7", use_cuda=use_cuda)

    def __enter__(self):
        assert os.path.isfile(self.video_in_path), "Error: path error"
        self.vdo.open(self.video_in_path)

        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.video_out_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print((self.im_width, self.im_height))
            self.output = cv2.VideoWriter(self.video_out_path, fourcc, 20, (self.im_width, self.im_height))
        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        while self.vdo.grab():
            start = time.time()
            _, im = self.vdo.retrieve()
            bbox_xcycwh, cls_conf, cls_ids = self.frcnn.detect(im)
            if bbox_xcycwh.any():
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # im = draw_bboxes(im, bbox_xyxy, identities)
            end = time.time()
            # print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
            if self.display:
                cv2.imshow("test", im)
                cv2.waitKey(1)
            if self.video_out_path:
                self.output.write(im)
            # exit(0)

class KjnDetectorImageFolder(object):
    def __init__(self, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.frcnn = KjnFastRCNN()
        self.deepsort = DeepSort("object_detection_and_tracing/deep_sort/deep/checkpoint/ckpt.t7", use_cuda=self.use_cuda)

    def detect_and_track_through_images(self, input_folder, bbox_output_folder = 'bboxes', output_folder = None):
        images = sorted(self._load_images(input_folder))
        output_raport = {
            "input_folder": input_folder,
            "bbox_output_folder": bbox_output_folder,
            "output_folder": output_folder,
            "raw_cut_images": images,
            "raw_cut_images_lenght": len(images),
            "single_image_dict": [],
        }
        for image_path in images:
            print(image_path)
            img = cv2.imread(image_path)
            print("img.shape: ", img.shape)
            start = time.time()
            bbox_xcycwh, cls_conf, cls_ids = self.frcnn.detect(img)
            print("object_detection_time: ", time.time() - start)
            if bbox_xcycwh.any():
                image_dict = {}
                start = time.time()
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, img)
                print("traking_time: ", time.time() - start)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    bbox_xyxy_list = bbox_xyxy.tolist()
                    image_dict.update({"bbox_xyxy": bbox_xyxy_list})
                    identities = outputs[:, -1]
                    identities_list = identities.tolist()
                    image_dict.update({"identities": identities_list})
                    if output_folder != None:
                        img_name = pathlib.Path(image_path).name
                        new_img_parh = os.path.join(output_folder, img_name)
                        img = self._draw_bboxes(img, bbox_xyxy, identities)
                        cv2.imwrite(new_img_parh, img)
                        image_dict.update({"img_with_bboxdraw_output_path": new_img_parh})

                    bboxes_save_paths = []
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        id = int(identities[i])
                        bbox_id_folder = os.path.join(bbox_output_folder, str(id))
                        pathlib.Path(bbox_id_folder).mkdir(parents=True, exist_ok=True)
                        bbox = img[y1:y2, x1:x2]
                        img_name = pathlib.Path(image_path).stem
                        bbox_save_path = os.path.join(bbox_id_folder, img_name+'.jpg')
                        cv2.imwrite(bbox_save_path, bbox)
                        bboxes_save_paths.append(bbox_save_path)
                    image_dict.update({"bboxes_save_paths": bboxes_save_paths})               
                output_raport['single_image_dict'].append(image_dict)
        return output_raport

    def _load_images(self, path):
        images = []
        valid_images = ['.png', '.jpg', '.jpeg']
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(os.path.join(path, f))
        return images

    def _draw_bboxes(self, img, bbox, identities=None, offset=(0,0)):
        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = (0, 255, 0)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 4 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        return img

if __name__ == "__main__":
    # import pathlib
    # input_folder = 'E:/kjn_biedronka/dataset_movies2/'
    # outpur_folder = 'E:/kjn_biedronka/dataset_movies2_tracking/'
    # for movie_path in load_movies(input_folder):
    #     print("movie_path: ", movie_path)
    #     new_movie_path = pathlib.Path(movie_path).stem
    #     print(new_movie_path+'.avi')
    #     new_movie_path = pathlib.Path(outpur_folder).joinpath(new_movie_path+'.avi')
    #     print(new_movie_path)
    #     with KjnDetector(movie_path, new_movie_path) as det:
    #         det.detect()
    #     break
    import pathlib
    input_folder = 'E:/kjn_biedronka/detectron2-deepsort-pytorch-master/cut_input_images'
    outpur_folder = 'E:/kjn_biedronka/detectron2-deepsort-pytorch-master/cut_output_images'
    kjn = KjnDetectorImageFolder()
    kjn.detect_and_track_through_images(input_folder, outpur_folder)
