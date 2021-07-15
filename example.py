import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import pyautogui as pgui
from queue import Queue as Q
from threading import Thread
# from pocketsphinx.pocketsphinx import *
# from sphinxbase.sphinxbase import *
# import pyaudio
from pynput.mouse import Button, Controller
torch.cuda.device("cuda")
class Ubimouse:
    def __init__(self):
        self.q = Q()
        self.quitQ = Q()
        self.model = []
        self.mean = []
        self.std = []
        self.inference_screen_width = 1920
        self.inference_screen_height = 1080
        self.training_screen_width = 1920
        self.training_screen_height = 1080

    def load_learner(self, file_path, file='export.pkl', test=None, tfm_y=None, **db_kwargs):
        self.model = []
        self.mean = []
        self.std = []
        """Load a `Learner` object saved with `export_state` in `path/file` with empty data, 
        optionally add `test` and load on `cpu`. `file` can be file-like (file or buffer) - FastAI"""
        source = file_path + '/' + file
        state = torch.load(source, map_location='cpu') if not torch.cuda.is_available() else torch.load(source)
        self.model = state.pop('model')
        try:
            self.mean = state['data']['normalize']['mean']
            self.std = state['data']['normalize']['std']
        except Exception as e:
            print(e)
        # model.load_state_dict(state, strict=True)
        return self.model, self.mean, self.std

    def fastai_loader_inference(self, file_path, file_name):
        net, mean, std = self.load_learner(file_path, file_name)
        """Get These mean and standard from FastAI model"""
        if len(mean) != 0 and len(std) != 0:
            photo_mean = mean
            photo_std = std
        else:
            photo_mean = torch.tensor(np.array([0.485, 0.456, 0.406]))
            photo_std = torch.tensor(np.array([0.229, 0.224, 0.225]))
        """This will make photos into Tensor and normalize them"""
        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(photo_mean, photo_std)
        ])
        return net, transform_norm

    # def Voice_Click(self):
    #     modeldir = "C:/Users/Benjamin/Desktop/sphinx/pocketsphinx/model"
    #     datadir = "C:/Python36/Lib/site-packages/pocketsphinx/data"
    #
    #     config = Decoder.default_config()
    #     config.set_string('-hmm', "C:/Users/Benjamin/Desktop/sphinx/pocketsphinx/model/en-us/en-us-v1-adapt")
    #     config.set_string('-dict', "C:/Users/Benjamin/Desktop/sphinx/pocketsphinx/model/cmudict-en-us.dict")
    #     config.set_string('-keyphrase', 'click')
    #     config.set_float('-kws_threshold', 1e-3)
    #
    #
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    #     stream.start_stream()
    #
    #     decoder = Decoder(config)
    #     decoder.start_utt()
    #
    #     while True:
    #         if not self.quitQ.empty():
    #             break
    #
    #         buf = stream.read(1024)
    #         if buf:
    #              decoder.process_raw(buf, False, False)
    #         else:
    #              break
    #         if decoder.hyp() != None:
    #             self.q.put(1)
    #             decoder.end_utt()
    #             decoder.start_utt()

    def live_video(self):
        DEVICE = 'cpu'
        if torch.cuda.is_available():
            DEVICE = 'cuda'

        """Start getting UbiMouse model and its setting /"""
        # Write Kees-type Model's path and filename
        file_path = r'C:\Users\chino_82i56t9\OneDrive\ドキュメント\Python Scripts\Ubimouse_kees\LatestModel\20200402'
        file_name = '31_03_34_resize1.pkl'
        net, transform_norm = self.fastai_loader_inference(file_path, file_name)  # 2nd one will be used later
        net.eval()
        net = net.to(DEVICE)
        """/ End getting UbiMouse model and its setting"""

        """Start getting Click model and its setting /"""
        # Write Kees-type Model's path and filename
        file_path = r'C:\Users\chino_82i56t9\OneDrive\ドキュメント\Python Scripts\Ubimouse_kees'
        file_name = '24_03_Click.pkl'
        net1, transform_norm1 = self.fastai_loader_inference(file_path, file_name)  # 2nd one will be used later
        net1.eval()
        net1 = net1.to(DEVICE)
        """/ End getting Click model and its setting"""

        pgui.FAILSAFE = False

        """Video Capture Setting"""
        video_capture = cv2.VideoCapture(1)
        try:
            ret = video_capture.set(3, 640)
            ret = video_capture.set(4, 480)
        except:
            print("Video size problem")
        print("width_set = " + str(video_capture.get(3)) + ", height_set = " + str(video_capture.get(4)))
        """This video size needs to be VGA since FastAI has used VGA when training"""
        resize = False
        if video_capture.get(3) != 640:
            resize = True

        """Screen Size of Display"""
        width = self.inference_screen_width
        height = self.inference_screen_height

        prev_xRatio = None
        prev_yRatio = None

        # xOffset = -5.0 / 320   #offset in pixels = xOffset * 320 + 320
        # yOffset = 10.0 / 240   #offset in pixels = uOffset * 240 + 240
        xOffset = 0 / 320   #offset in pixels = xOffset * 320 + 320
        yOffset = 0 / 240   #offset in pixels = uOffset * 240 + 240

        # pixelScaleX = 2.0 #amount of scaling needed so that all the edges are reached (sensitivity)
        # pixelScaleY = 2.3 #amount of scaling needed so that all the edges are reached (sensitivity)
        pixelScaleX = 1.7 #amount of scaling needed so that all the edges are reached (sensitivity)
        pixelScaleY = 1.7 #amount of scaling needed so that all the edges are reached (sensitivity)

        # xScale = width*pixelScale
        # yScale = height*pixelScale

        mouse = Controller()
            
        # buttonxLocation = 750
        # buttonyLocation = 550
        # speed = 10


        while True:
            with torch.no_grad():
                ret, frame = video_capture.read()
                if resize:
                    frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                cv2.imshow('Video', frame) 
                # frame = frame / 256 - photo_mean / photo_std + 1
                start_time = time.time()

                # # img_tensor = torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2, 0, 1) / 255).float()
                # # img_tensor = torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2, 0, 1)).float()
                # img_tensor = torch.tensor(np.ascontiguousarray(np.flip(frame, 2) / 128 - photo_mean / photo_std).transpose(2, 0, 1)).float()
                # # img_tensor = torch.tensor(np.ascontiguousarray(np.flip(frame, 2) / 256 - photo_mean / photo_std).transpose(2, 0, 1)).float()

                """Start image preprocessing /"""
                """OpenCV => PIL conversion for the model"""
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                """To Tensor with normalization"""
                img_tensor = transform_norm(frame)
                """Magic words :)"""
                img_tensor = img_tensor.unsqueeze(0)
                """Bring image tensor to GPU or CPU"""
                img_tensor = img_tensor.to(DEVICE)
                """/ End image preprocessing"""

                """Give image tensor to the model for processing"""
                p = net(img_tensor)
                p1 = net1(img_tensor)
                print(
                    f"""
                    #####
                    ({p[0][0]},{p[0][1]})
                    {p1[0][0]}
                    #####
                    """
                )

                print("--- %s seconds ---" % (time.time() - start_time))
                if not self.q.empty():
                    self.q.get()
                    mouse.click(Button.left, 1)
                    time.sleep(0.0)

                xRatio = (p.data[0][1] * 320 + 320)/self.training_screen_width  # ratio converted from the 640 by 480 scale to 1920 by 1080 between 0 and 1
                yRatio = (p.data[0][0] * 240 + 240)/self.training_screen_height  # ratio converted from the 640 by 480 scale to 1920 by 1080 between 0 and 1

                xRation = ((xRatio-0.5-xOffset)*pixelScaleX) + 0.5  # convert to -0.5 and 0.5 then apply the pixelScale and then scale back to 0 and 1
                yRation = ((yRatio-0.5-yOffset)*pixelScaleY) + 0.5  # convert to -0.5 and 0.5 then apply the pixelScale and then scale back to 0 and 1
             
                xRationi = (xRation*width)
                yRationi = (yRation*height)
                
                if prev_xRatio is not None:
                    deltax = prev_xRatio-xRationi
                    deltay = prev_yRatio-yRationi  
                    xRationi = xRationi + (0.7*deltax)
                    yRationi = yRationi + (0.7*deltay)
                    if (((xRationi-prev_xRatio)**2+(yRationi-prev_yRatio)**2) > 14**2):
                        mouse.position = (xRationi), (yRationi)                                               #Time = 0.0 s
    # ---------------------------------------------------------------------------------------------------------------------------
                prev_xRatio = xRationi
                prev_yRatio = yRationi                                                                        #Time = 0.0 s        
    # ---------------------------------------------------------------------------------------------------------------------------
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quitQ.put(1)
                    break
        video_capture.release()
        cv2.destroyAllWindows()


gd = Ubimouse()
# dc = gd.Voice_Click
cf = gd.live_video
# t1 = Thread(target = dc)
t2 = Thread(target=cf)

t2.start()
# t1.start()
