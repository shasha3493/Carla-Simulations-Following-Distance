# Wistron Neweb Corporation 🚀, AGPL-3.0 license

import torch
import onnxruntime
import cv2
import numpy as np

from utils import *


class Model:
    def __init__(self, model, conf=0.5):
        """
        Initialize Model backend class

        Args:
            model (Union[str, Path]): Path or name of the model to load or create
            conf (float): Confidence threshold for object detection task
        """

        assert 0 < conf < 1, f"'conf' must be between 0 and 1, but 'conf'={conf}"
        self.conf = conf
        self.done_warmup = False

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Currently using device: {self.device}")
        self.setup_model(weight=model)

    def setup_model(self, weight):
        "Set up model with given weight"

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(weight, providers=providers)
        metadata = self.session.get_modelmeta().custom_metadata_map

        if metadata:
            for k, v in metadata.items():
                if k in ('stride', 'batch'):
                    metadata[k] = int(v)
                elif k in ('imgsz', 'names_det', 'names_lane', 'names_drive') and isinstance(v, str):
                    metadata[k] = eval(v)
        
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.names_det = metadata['names_det']
        self.names_drive = metadata['names_drive']
        self.names_lane = metadata['names_lane']
        self.imgsz = metadata['imgsz']

    def warmup(self, imgsz):
        im = torch.empty(imgsz, dtype=torch.float, device=self.device)
        self.done_warmup = True
        for _ in range(1):
            self.forward(im)
    
    def forward(self, im):
        im = im.cpu().numpy()
        y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        return self.from_numpy(y)
    
    def from_numpy(self, x):
        return [torch.tensor(y).to(self.device) if isinstance(y, np.ndarray) else y for y in x]

    @smart_inference_mode()
    def __call__(self, im=None):
        if not self.done_warmup:
            self.warmup(imgsz=(1, 3, *self.imgsz))
        im = self.preprocess(im)
        preds = self.forward(im)
        return self.postprocess(preds)
    
    def preprocess(self, im):
        "Prepare input image for inference"
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            self.orig_imgsz = im.shape[:2]
        if not_tensor:
            if im.shape[:2] != self.imgsz:
                im = cv2.resize(im, self.imgsz[::-1], interpolation=cv2.INTER_LINEAR)
            im = im.transpose((2, 0, 1))[::-1, :, :]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)
        if len(im.shape) == 3:
            im = torch.unsqueeze(im, 0)
        im = im.to(self.device)
        im = im.float()
        if not_tensor:
            im /= 255
        return im
    
    def postprocess(self, preds):
        "Post processing"
        lane, drive, det = preds

        det = non_max_suppression(
            det, conf_thres=self.conf, iou_thres=0.5)[0].detach()
        # for box in det:
        #     box[:4] = scale_boxes(self.imgsz, box[:4], self.orig_imgsz)
        lane = torch.squeeze(torch.argmax(lane, dim=1)).detach().cpu().numpy()
        drive = torch.squeeze(torch.argmax(drive, dim=1)).detach().cpu().numpy()
        return (lane, drive, det)

