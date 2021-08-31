# import cv2
# from fastai import *
# from fastai.vision import *
# import fastai
# import numpy as np
import torch
# In case you got PosixPth problem on windows, add these lines
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_learner(file_path, file='export.pkl', is_fastaiv2=False, test=None, tfm_y=None, **db_kwargs):
    model = []
    mean = []
    std = []
    """Load a `Learner` object saved with `export_state` in `path/file` with empty data, 
    optionally add `test` and load on `cpu`. `file` can be file-like (file or buffer) - FastAI"""

    source = file_path + '/' + file
    state = torch.load(source, map_location='cpu') if not torch.cuda.is_available() else torch.load(source)
    if is_fastaiv2:
        model = state.model
    else:  # FastAI_v1
        model = state.pop('model')
        try:
            mean = state['data']['normalize']['mean']
            std = state['data']['normalize']['std']
        except Exception as e:
            print(e)
        # model.load_state_dict(state, strict=True)
    return model, mean, std


mnist = r'Z:\PythonScripts\ModelConversionTest'
# mnist = r'\\chinou-disk-02\disk1\tmp\UbiMouse_DataSet\DataForViewsonicMonitor'
# mnist1 = r'C:\\Users\\chino_82i56t9\\OneDrive\\ドキュメント\\Python Scripts\\Ubimouse_kees'

model, mean, std = load_learner(mnist, 'OpenedClosedModel-ResNet50.pkl', is_fastaiv2=True)
# learn = load_learner(mnist, '15MAR2021_viewsonic.pkl')
# learn1 = load_learner(mnist1, '24_03_Click.pkl')

model.eval()
# learn1.model.eval()

# x = torch.randn(1, 3, 480, 640, requires_grad=False)  #.cuda()
x = torch.randn(1, 3, 240, 320, requires_grad=False)  #.cuda()
torch_out = torch.onnx._export(model, x, r'Z:\PythonScripts\ModelConversionTest\OpenedClosedModel-ResNet50.onnx', export_params=True)
# x1 = torch.randn(1, 3, 480, 640, requires_grad=False)  #.cuda()
# torch_out1 = torch.onnx._export(learn1.model, x1, r'C:\\Users\\chino_82i56t9\\OneDrive\\ドキュメント\\Python Scripts\\Ubimouse_kees\\test_learn1.onnx', export_params=True)
with open("model_data_mean_std.json", "w") as f:
    txt_lines = f'"mean": {mean}, "std": {std}'
    f.write(txt_lines)
print("DONE")
