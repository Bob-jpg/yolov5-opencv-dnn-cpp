# yolov5-opencv-dnn-cpp
使用opencv模块部署yolov5-6.0版本

基于6.0版本的yolov5:https://github.com/ultralytics/yolov5

**OpenCV>=4.5.0**

+ 导出onnx模型需要将opset设置成12（原来默认的是13，在opencv下面会报错，原因未知）</br>
+ 如果是torch1.12.x的版本,需要在
https://github.com/ultralytics/yolov5/blob/c98128fe71a8676037a0605ab389c7473c743d07/export.py#L155
这里的```do_constant_folding=False```,设置为false才行，否者读取网络会失败，原因未知。<br>
```
$ python path/to/export.py --weights yolov5s.pt --img [640,640] --opset 12 --include onnx
```
#### 2022.12.13 更新：
+ 如果你的显卡支持FP16推理的话，可以将模型读取代码中的```DNN_TARGET_CUDA```改成```DNN_TARGET_CUDA_FP16```提升推理速度（虽然是蚊子腿，好歹也是肉（： 
#### 2022.03.29 更新：  

+ 新增P6模型支持，可以通过yolo.h中定义的YOLO_P6切换  

+ 另外关于换行符，windows下面需要设置为CRLF，上传到github会自动切换成LF，windows下面切换一下即可

以下图片为更新p6模型之后yolov5s6.onnx运行结果：
![zidane](https://user-images.githubusercontent.com/52729998/160559827-45572f7e-54e8-4653-b9be-6d287912b065.jpg)

![bus](https://user-images.githubusercontent.com/52729998/160559831-3ddf926d-b7c3-4687-bd57-26dd4d1cc055.jpg)



#### 1.yolov5-6.0模型训练注意事项
+ 利用gpu进行训练时会出现torch==false,,,,,可按照该https://blog.csdn.net/qq_42709514/article/details/121168753 
+ 当出现using waring cuda0和cuda1的警告时，需要修改的参数
+ workers==0       
+ batchsize==8
```
torch_utils.py 中if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        print(torch.cuda.is_available())        ########################修改此处
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
```
#### 2.训练完成后进行检测


#### 3.进行模型转化pt--onnx时，需要修改代码,同时需要对模型进行简化处理
```
 parser.add_argument('--include', nargs='+',
                        default=['torchscript'，'onnx'],
                        help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
修改为
parser.add_argument(
        '--include',
        nargs='+',
        default=['onnx'],
        help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle')
```
```
python path/to/export.py --weights yolov5s.pt --img [640,640] --opset 12 --include onnx
```

```
###简化模型
pip install onnx-simplifier
python -m onnxsim input_onnx_model output_onnx_model
```

### 4.yolov5-opencv-dnn模型部署
+ 仿照胡工写的dointerfence进行修改，注意事项
+ (1)
```
如果opencv=4.6时候，此处需要补充
std::sort(netOutputImg.begin(), netOutputImg.end(), [](Mat& A, Mat& B) {return A.size[1] > B.size[1]; });
```
+ (2)
```
当用自己的模型推理时，需要重新封装，将此处修改为自己标签名
std::vector<std::string> classes = { "0", "1" };
```
+ (3)测试自己封装的第一层的dll时，把动态链接库直接转化为windows窗口控制台应用直接运行
+ (4)测试自己训练模型需要对模型进行简化（onnx）处理
+ (5)二次封装时需要保证调用dll时候的函数名一致


### 5.移植工控机上
+ (1)安装visual studio
+ (2)当出现无法加载dll时，需要将opencv依赖的dll放置windows/systym32/文件下
