# Introduction

ZQCNN is an inference framework that can run under windows, linux and arm-linux. At the same time, there are some demos related to face detection and recognition.

## Main development environment: [VS2015 with Update 3](https://pan.baidu.com/s/1zoREccOxVsggV-iI2z4HTg)

  MKL download address: [Download here](https://pan.baidu.com/s/1d75IIf6fgTZ5oeumd0vtTw)

## Core module supports linux:

  If it cannot be fully compiled according to [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and others for you program you want to test.

## Core module supports arm-linux:

  If it cannot be fully compiled according to [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and others for you program you want to test.
  
**BUG:** cmake .. -DSIMD_ARCH_TYPE=arm64 -DBLAS_TYPE=openblas_zq_gemm

Ideally one would use the faster of openblas and ZQ_GEMM to calculate the convolution (I chose the branch by testing the time on cortex-A72). However, this option currently does not achieve the expected results.
  Need to manually note the definition in ZQ_CNN_CompileConfig.h
  
```
#define ZQ_CNN_USE_ZQ_GEMM 1
#define ZQ_CNN_USE_BLAS_GEMM 1
```
	
Can be commented out
  
```
line 67: #if defined(ZQ_CNN_USE_BOTH_BLAS_ZQ_GEMM)
line 70: #endif
```

## Training related

  pytorch training SSD: https://github.com/zuoqing1988/pytorch-ssd-for-ZQCNN

  Training gender age: https://github.com/zuoqing1988/train-GenderAge
	
  Training MTCNN: https://github.com/zuoqing1988/train-mtcnn
	
  Training SSD: https://github.com/zuoqing1988/train-ssd
	
  Training MTCNN for head detection: https://github.com/zuoqing1988/train-mtcnn-head


# Update log

**Update on August 18, 2022: Optimized the 106-point pipeline for faces in video mode, and added a new head posture and line of sight model**

The demo program is in SampleVideoFaceDetection_Interface.cpp

The original 106-point pb model and the head pose and line of sight pb model are in TensorFlow_to_ZQCNN

**Updated on April 20, 2022: Supports the SSD model trained by pytorch-ssd-for-ZQCNN**

**Updated on 2020-05-08: Added text recognition example SampleOCR**

- There is no text detection capability yet, and the image that needs to be input is cropped.
	
- Model download address:
	
  Link: https://pan.baidu.com/s/1O75LRBjXWwPXqAshLMJV3w Extraction code: f2q8
	
**Updated on March 22, 2020: Provides a set of MTCNN models that can detect faces with masks**

```
model\det1-dw20-plus.zqparams
model\det1-dw20-plus.nchwbin

model\det2-dw24-p0.zqparams
model\det2-dw24-p0.nchwbin
	
model\det3-dw48-p0.zqparams
model\det3-dw48-p0.nchwbin
```

**Updated on 2019-07-08: ZQCNN model to MNN model code**

[Click here to read](https://github.com/zuoqing1988/ZQCNN/tree/master/ZQCNN_to_MNN)

**Update on May 28, 2019: Open source a quasi-commercial grade 106-point model**

ZQCNN format: in the model folder det5-dw112

mxnet format: Link: https://pan.baidu.com/s/19DTG3rmkct8AiEu0l3DYjw Extraction code: qjzk


**Update on March 16, 2019: Reached 800 stars, announced a more accurate 106-point landmark model**

[ZQCNN format: det5-dw96-v2s](https://github.com/zuoqing1988/ZQCNN/tree/master/model) det5-dw96-v2s.zqparams, det5-dw96-v2s.nchwbin in the model folder

[mxnet format: Lnet106_96_v2s](https://pan.baidu.com/s/1iuuAHgJBsdWsUoAdU5H58Q) Extraction code: r5h2

**Update on 2019-02-14: Reached 700 stars, announced selected models for face detection**

[ZQCNN format: Selected 6 types of Pnet, 2 types of Rnet, 2 types of Onet, 2 types of Lnet](https://pan.baidu.com/s/1X2U9Y-6MJw3md8WuYxaotw)



| Six types of Pnet | Input size | Calculation amount (excluding bbox) | Remarks |
| -------- | ------ | ------------ | -------------------- |
| [Pnet20_v00](https://pan.baidu.com/s/1g7JnOxnbXIbNWPXGI-IzrQ) | 320x240 | 8.5 M | Benchmark libfacedetection |
| [Pnet20_v0](https://pan.baidu.com/s/1r3VcmEX1a2C5gKlGKnC4kw) | 320x240 | 11.6 M | Benchmark libfacedetection |
| [Pnet20_v1](https://pan.baidu.com/s/1qVU3_nporbOUzXYu7giZkA) | 320x240 | 14.6 M | |
| [Pnet20_v2](https://pan.baidu.com/s/1bXzdmsTgfqU_TJHsozSmrQ) | 320x240 | 18.4 M | Benchmark original pnet |
| [Pnet16_v0](https://pan.baidu.com/s/1s5eZLeAKnqp1ZDTrzaOD_w) | 256x192 | 7.5 M | stride=4 |
| [Pnet16_v1](https://pan.baidu.com/s/1Lf0z6rRq5WUKE_DMze_C7w) | 256x192 | 9.8 M | stride=4 |

| Two Rnets | Input size | Calculation amount | Remarks |
| -------- | ------ | ------------ | -------------------- |
| [Rnet_v1](https://pan.baidu.com/s/1SEIolnvmtPvdqbHxU1vPWQ) | 24x24 | 0.5 M | Benchmark original Rnet |
| [Rnet_v2](https://pan.baidu.com/s/1APWYGcFC5MAn6Ba5vWo80w) | 24x24 | 1.4 M | |

| Two types of Onet | Input size | Calculated amount | Remarks |
| -------- | ------ | ------------ | -------------------- |
| [Onet_v1](https://pan.baidu.com/s/1UTvSKErOul2wkT5EMxXgVA) | 48x48 | 2.0 M | Without landmark |
| [Onet_v2](https://pan.baidu.com/s/19QomSIy3Py516OEIBFDcVg) | 48x48 | 3.2 M | Without landmark |

| Two types of Lnet | Input size | Calculation amount | Remarks |
| -------- | ------ | ------------ | -------------------- |
| [Lnet_v2](https://pan.baidu.com/s/1W6bxNeD0psxwxbou_xwK-g) | 48x48 | 3.5 M | lnet_basenum=16 |
| [Lnet_v2](https://pan.baidu.com/s/1e3tuwrR3AoU_zRKkIFK8xg) | 48x48 | 10.8 M | lnet_basenum=32 |

**Updated on January 31, 2019: Reached 600 stars, announced the MTCNN head detection model**

Trained with 'hollywoodheads' data, the effect is average, just use it.

Head detection mtcnn-head[mxnet-v0](https://pan.baidu.com/s/11I-ZnW3AAijlijtroyxClQ)&[zqcnn-v0](https://pan.baidu.com/s/1Xh27qm_LmuV6ZIDLBUXfPQ)



**Updated on January 24, 2019: The core module supports linux**

If it cannot be fully compiled according to [build-with-cmake.md](https://github.com/zuoqing1988/ZQCNN/blob/master/build-with-cmake.md), you can only compile ZQ_GEMM, ZQCNN, and others for you program you want to test.

**Updated on 2019-01-17**

Changed ZQ_CNN_MTCNN.h

(1) When thread_num is set to less than 1 during init, Pnet_stage can be forced to execute multi-threading, which means it will be divided into blocks. This can prevent the memory from exploding when looking for small faces in big pictures.

(2) The size of rnet/onet/lnet does not need to be 24/48/48, but only supports equal width and height.

(3) rnet/onet/lnet is processed in batches, which can reduce memory usage when there are many faces.

**Update on January 15, 2019: Celebrate reaching 500 stars and distribute 106 landmark models**

[mxnet format & zqcnn format](https://pan.baidu.com/s/18VTMfChnAEyeU_9vE9GJaw)


**Update on January 4, 2019: Celebrate reaching 400 stars and distribute fast face models**

[mxnet format](https://pan.baidu.com/s/1pOvAaXncbarNfD0G-4BwlQ)

[zqcnn format](https://pan.baidu.com/s/18FLOduY4SoHjXHBCXWQ5LQ)

The v3 version is not good enough, and a v4 version will be released later, which is probably what the picture below means.

![MTCNN-v4 diagram](https://github.com/zuoqing1988/ZQCNN/blob/master/mtcnn%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)

**~~Updated on 2018-12-25: 106 points of landmark that are not open source~~**

~~Life is relatively tight, so I want to earn some extra money.~~

~~landmark106-normal-1000.jpg is the landmark generated by model\det5-dw48-1000.nchwbin~~
	
~~landmark106-normal.jpg and landmark106-big.jpg are the two models I trained that are not open source~~
	
~~The normal model is 2.1M, the calculation amount is 11.4M, and the PC single thread takes 0.6-0.7ms. The big model is 7.56M, the calculation amount is 36.4M, and the PC single thread takes 1.5-1.6ms~~

**Updated on 2018-12-20: Added MTCNN 106-point landmark model**

Try it out in SampleMTCNN (what’s released is just a not so good one, and better ones are waiting to be sold for money)

SampleLnet106 has timing, about 0.6~0.7ms for a single thread (E5-1650V4, 3.6GHz)

**Updated on 2018-12-03: Compile the model into the code**

model2code in ZQCNN.sln can compile the model into code

```
model2code.exe param_file model_file code_file prefix
```
	
Then add it to your project

```
#include"code_file"
```
	
Use the following function to load the model

```
LoadFromBuffer(prefix_param, prefix_param_len, prefix_model, prefix_model_len)
```


**Updated on 2018-11-21**

For models that support mxnet-ssd training, mean_val needs to be set to 127.5 to run correctly in SampleSSD.

But it seems that the training using ReLU is not correct. I used PReLU to train one and started training again. It only has mAP=0.48. Let’s make do with it. [Click here to download](https://pan.baidu.com/s/1-wfpuvGLBGPtlqicdO1raw) .

After changing the model, you must use imagenet to first train the classification model, and then train the SSD to increase the mAP.

**Updated on 2018-11-14**

(1) Optimize ZQ_GEMM. On a 3.6GHz machine, the MKL peak is about 46GFLOPS and ZQ_GEMM is about 32GFLOPS. The overall time using the ZQ_GEMM face model is about 1.5 times that of using MKL.

Note: ZQ_GEMM compiled using VS2017 is faster than VS2015, but SampleMTCNN multi-threaded operation is wrong (maybe OpenMP support rules are different?).

(2) Very small weights can be removed when loading the model. When you find that the model is much slower than expected, it is probably because the weight values ​​are too small.

**Updated on 2018-11-06**

(1) Remove all omp multi-threaded code in layers. The calculation amount is too small and the speed is slower than single thread.

(2) cblas_gemm can choose MKL, but the mkl provided by 3rdparty is very slow on my machine, and the dll is relatively large. I did not put it in 3rdparty\bin. Please download it from [here](https://pan.baidu.com /s/1d75IIf6fgTZ5oeumd0vtTw).

**Update 2 on October 30, 2018: It is recommended to use Gaussian filter first to find small faces in MTCNN big pictures**

**Updated on 2018-10-30: BatchNorm eps problem**

(1)The default eps of BatchNorm and BatchNormScale are both 0

(2) If you use mxnet2zqcnn to transfer the model from mxnet, eps will be added to var as a new var during the transfer process.

(3) If the model is transferred from other platforms, either manually add eps to var, or add eps=? (? is the eps value of this layer of the platform) after BatchNorm and BatchNormScale.

Note: In order to prevent division by 0 errors, when dividing var, it is calculated as sqrt(__max(var+eps,1e-32)). That is to say, if var+eps is less than 1e-32, it will be slightly different from the theoretical value. .
However, after today's modification, the LFW accuracy of the following face models is exactly the same as the results of minicaffe.

**Updated on 2018-10-26**

MTCNN supports multi-threading. When looking for small faces in a large picture and there are many faces, 8 threads can achieve more than 4 times the effect of single thread. Please use data\test2.jpg to test

**Updated on 2018-10-15**

Improve the nms strategy of MTCNN: 1. The local maximum of the nms of the Pnet of each scale must cover a certain number of non-maximums, and the number is set in the parameters; 2. When the resolution of the Pnet is too large, the nms is processed in blocks .

**Updated on 2018-09-25**

GNAP supports insightface and automatically converts models using mxnet2zqcnn, see [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn). You can try [MobileFaceNet-GNAP](https://pan.baidu.com/s/1hv4lbYwSLlLiGK07FuJM5Q)

**Updated on 2018-09-20**

(1) To update the test method for tar-far accuracy of the face recognition model, you can follow the steps [How-to-evaluate-TAR-FAR-on-your-dataset](https://github.com/zuoqing1988/ZQCNN-v0 .0/wiki/How-to-evaluate-TAR-FAR-on-your-dataset) Construct a test set yourself to test the model accuracy.

(2) According to (1) I cleaned CASIA-Webface and constructed two test sets [webface1000X50](https://pan.baidu.com/s/1AoJkj_IhydkiyD1UGm8rDQ), [webface5000X20](https://pan.baidu.com /s/1AoJkj_IhydkiyD1UGm8rDQ), and tested the accuracy of several major face recognition models that I open sourced.

**Updated on 2018-09-13**

(1) Support loading models from memory

(2) Add the compilation configuration ZQ_CNN_CompileConfig.h, you can choose whether to use _mm_fmadd_ps, _mm256_fmadd_ps (you can test the speed to see if it is faster or slower).

**2018-09-12 update using [insightface](https://github.com/deepinsight/insightface) to train 112*96 (that is, the size of sphereface) steps: ** [InsightFace: how to train 112*96] (https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/InsightFace%EF%BC%9A-how-to-train-112*96)

**Updated on 2018-08-15**

(1) Add natural scene text detection, the model is transferred from [TextBoxes](https://github.com/MhLiao/TextBoxes). Personally I feel it's too slow and not very accurate.

Note that the PriorBoxLayer used in this project is different from the PriorBoxLayer in SSD. In order to export the weights in ZQCNN format, I modified deploy.prototxt and saved it as deploy_tmp.prototxt.
Download the model from [here](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ).

(2) Add pictures to detect pornography. The model is transferred from [open_nsfw](https://github.com/yahoo/open_nsfw). I have not tested whether the accuracy is high or not.

Download the model from [here](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw).

**Updated on 2018-08-10**

Successfully transferred [GenderAge-r50 model](https://pan.baidu.com/s/1f8RyNuQd7hl2ItlV-ibBNQ) and [Arcface-LResNet100E-IR](https://pan.baidu.com/s/ on mxnet 1wuRTf2YIsKt76TxFufsRNA), the steps are the same as converting to MobileFaceNet model.
Check out [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

The Model Zoo below contains the model I converted, which should be slightly faster than the automatic conversion.

Open ZQCNN.sln and run SampleGenderAge to see the effect. On my E5-1650V4 CPU, the single-thread time fluctuates greatly, with an average of about 1900-2000ms and four-threads of more than 400 ms.

**Updated on 2018-08-09**

Added mxnet2zqcnn, successfully converted MobileFaceNet on mxnet into ZQCNN format (there is no guarantee that other models can be converted successfully, ZQCNN does not support many Layers yet). Check out [mxnet2zqcnn](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/mxnet2zqcnn)

**Updated on 2018-08-07**

BUG fix: Convolution, DepthwiseConvolution, InnerProduct, BatchNormScale/Scale previously defaulted to with_bias=true, but now the default is with_bias=false. That is to say, the previous code cannot load these layers without bias.

For example, a Layer like the following used to have bias_term by default, but now it does not have bias_term by default.

Convolution name=conv1 bottom=data top=conv1 num_output=10 kernel_size=3 stride=1

**Updated on 2018-08-06**

Added face recognition accuracy test in LFW database. Open ZQlibFaceID.sln to see the related Project.

Since the calculation accuracy of C++ code is slightly different from that of matlab, the statistical accuracy is also slightly different, but the difference is within 0.1%.

**Updated on 2018-08-03**

Supports multi-threading (accelerated via openmp). **Please note that currently multi-threading is slower than single-threading**

**Updated on 2018-07-26**

Support MobileNet-SSD. The model I use to convert caffemodel refers to export_mobilenet_SSD_caffemodel_to_nchw_binary.m. You need to compile matcaffe.
You can try this version [caffe-ZQ](https://github.com/zuoqing1988/caffe-ZQ)

**Updated on 2018-06-05**

Keep up with the trend of the times and release source code.
I forgot to mention that I need to rely on openblas. I directly used the version in mini-caffe, and it was very slow to compile it by myself.



# Model Zoo

**Face Detection**

[MTCNN-author-version](https://pan.baidu.com/s/1lWLKDYv8YQ6Th6KRiKvgug) Format transferred from [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment)

[MTCNN-ZQ-version](https://pan.baidu.com/s/1j1WqkwbUCf_9f4hCQukoFg)

**Face recognition (if not specified, the models are all trained by ms1m-refine-v2)**






|Model |LFW accuracy (ZQCNN) | LFW accuracy (OpenCV3.4.2) | LFW accuracy (minicaffe) |Time consuming (ZQCNN) |Remarks
|--------------------- |------------- |---------------------- | ---------------- |-------------------------- | -------------
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|99.67%-99.55%(matlab crop), 99.72-99.60%(C++ crop) |99.63 %-99.65%(matlab crop), 99.68-99.70%(C++ crop) |99.62%-99.65%(matlab crop), 99.68-99.60%(C++ crop)|The time is close to dim256|The network structure is the same as dim256, except Output dimensions are different
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|99.60%-99.60%(matlab crop), 99.62-99.62%(C++ crop) |99.73 %-99.68%(matlab crop), 99.78-99.68%(C++ crop) |99.55%-99.63%(matlab crop), 99.60-99.62%(C++ crop)|Single thread is about 21-22ms, four threads is about 11-12ms , 3.6GHz |The network structure is in the download link, trained with faces_emore
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|99.52%-99.60%(matlab crop), 99.63-99.72%(C++ crop) |99.70%-99.67%(matlab crop), 99.77-99.77%(C++ crop) |99.55%-99.62%(matlab crop), 99.62-99.68%(C++ crop)|The time is close to dim256|The network structure is the same as dim256, Only the output dimensions are different. Thanks to [moli](https://github.com/moli232777144) for training this model

|Model |LFW accuracy (ZQCNN) | LFW accuracy (OpenCV3.4.2) | LFW accuracy (minicaffe) |Time consuming (ZQCNN) |Remarks
|--------------------- |------------- |---------------------- | ---------------- |-------------------------- | -------------
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop) |99.82%-99.83%(matlab crop), 99.80-99.78%(C++ crop) |99.72%-99.72%(matlab crop), 99.72-99.68%(C++ crop)|The time is close to dim256|The network structure is the same as dim256, It’s just that the output dimensions are different
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|99.78%-99.78%(matlab crop), 99.75-99.75%(C++ crop) |99.82 %-99.82%(matlab crop), 99.80-99.82%(C++ crop) |99.78%-99.78%(matlab crop), 99.73-99.73%(C++ crop)|Single thread is about 32-33ms, four threads is about 16-19ms , 3.6GHz |The network structure is in the download link, trained with faces_emore
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|99.80%-99.73%(matlab crop), 99.85-99.83%(C++ crop) |99.83 %-99.82%(matlab crop), 99.87-99.83%(C++ crop) |99.80%-99.73%(matlab crop), 99.85-99.82%(C++ crop)|The time is close to dim256|The network structure is the same as dim256, except The output dimensions are different. Thanks to [moli](https://github.com/moli232777144) for training this model

|Model\test set webface1000X50 |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR =1e-5
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.78785 |9.274% |0.66616 |40.459% |0.45855 |92.716%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.77708 |7.839% |0.63872 |40.934% |0.43182 |92.605%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.76699 |8.197% |0.63452 |38.774% |0.41572 |93.000%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.79268 |9.626% |0.65770 |48.252% |0.45431 |95.576%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.76858 |9.220% |0.62852 |46.195% |0.40010 |96.929%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.76287 |9.296% |0.62555 |44.775% |0.39047 |97.347%

|Model\test set webface5000X20 |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR=1e-5|TAR@ FAR =1e-5
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.70933 |29.558% |0.51732 |85.160% |0.45108 |94.313%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.68897 |28.376% |0.48820 |85.278% |0.42386 |94.244%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.68126 |27.708% |0.47260 |85.840% |0.40727 |94.632%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.71238 |32.153% |0.51391 |89.525% |0.44667 |96.583%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.68490 |30.639% |0.46092 |91.900% |0.39198 |97.696%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.67303 |32.404% |0.45216 |92.453% |0.38344 |98.003%

|Model\test set TAO ids:6606,ims:87210 |thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR@ FAR=1e-6|thresh@ FAR= 1e-5|TAR@ FAR=1e-5
|--------------------- |------------- |-------------|-------- ------- |-------------| --------------- |-----------
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ)|0.92204 |01.282% |0.88107 |06.837% |0.78302 |41.740%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA)|0.91361 |01.275% |0.86750 |07.081% |0.76099 |42.188%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw)|0.90657 |01.448% |0.86061 |07.299% |0.75488 |41.956%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg)|0.92098 |01.347% |0.88233 |06.795% |0.78711 |41.856%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw)|0.90862 |01.376% |0.86397 |07.083% |0.75975 |42.430%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA)|0.90710 |01.353% |0.86190 |06.948% |0.75518 |42.241%

|Model\test set ZQCNN-Face_5000_X_20 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR @ FAR=1e-6
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[MobileFaceNet-GNAP](https://pan.baidu.com/s/1UL4Am0R2MYQOH6lZnPsvTg) |0.73537 |11.722% |0.69903 |20.110% |0.65734 |33.189%
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ) |0.64772 |40.527% |0.60485 |55.345% |0.55571 |70.986%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA) |0.61647 |42.046% |0.57561 |55.801% |0.52852 |70.622%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw) |0.59725 |44.651% |0.55690 |58.220% |0.51134 |72.294%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg) |0.64519 |47.735% |0.60247 |62.882% |0.55342 |77.777%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw) |0.58229 |56.977% |0.54582 |69.118% |0.49763 |82.161%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA) |0.58296 |54.731% |0.54219 |68.613% |0.49174 |82.812%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)|0.58058 |61.826% |0.53841 |75.281% |0.49098 |86.554%

|Model\test set ZQCNN-Face_5000_X_20 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6|TAR @ FAR=1e-6
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ)(not trained by myself) |0.61953 |47.103% |0.57375 |62.207% |0.52226 |76.758%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA) (ms1m-refine-v1 is not trained by myself) |0.61299 |50.594% |0.56658 |65.757% |0.51637 |79.207%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (not trained by myself) |0.57350 |67.434% |0.53136 |79.944% |0.48164 |90.147%


|Model\test set ZQCNN-Face_12000_X_10-40 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6 |TAR@ FAR=1e-6
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[MobileFaceNet-res2-6-10-2-dim128](https://pan.baidu.com/s/1AQEad5Zp2cag4UA5KtpbYQ) |0.64507 |39.100% |0.60347 |53.638% |0.55492 |69.516%
|[MobileFaceNet-res2-6-10-2-dim256](https://pan.baidu.com/s/143j7eULc2AqpNcSugFdTxA) |0.61589 |39.864% |0.57402 |54.179% |0.52596 |69.658%
|[MobileFaceNet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw) |0.60030 |41.309% |0.55806 |55.676% |0.50984 |70.979%
|[MobileFaceNet-res4-8-16-4-dim128](https://pan.baidu.com/s/1z6H5p4b3aVun2-1dZGDXkg) |0.64443 |45.764% |0.60060 |61.564% |0.55168 |76.776%
|[MobileFaceNet-res4-8-16-4-dim256](https://pan.baidu.com/s/1f_VtqNRxDNe972h8UrOsPw) |0.58879 |52.542% |0.54497 |67.597% |0.49547 |81.495%
|[MobileFaceNet-res4-8-16-4-dim512](https://pan.baidu.com/s/14ukmtAWDhIJC6312WBhZhA) |0.58492 |51.752% |0.54085 |67.104% |0.49010 |81.836%
|[MobileFaceNet-res8-16-32-8-dim512](https://pan.baidu.com/s/1On5BfcrOB5jrTrRD40vLkw)|0.58119 |61.412% |0.53700 |75.520% |0.48997 |86.647%

|Model\test set ZQCNN-Face_12000_X_10-40 |thresh@ FAR=1e-8|TAR@ FAR=1e-8|thresh@ FAR=1e-7|TAR@ FAR=1e-7|thresh@ FAR=1e-6 |TAR@ FAR=1e-6
|--------------------- | ------------- | ---------- |---------- ---- |------- | ------------ |-----------
|[ArcFace-r34-v2](https://pan.baidu.com/s/1q3ZqQdjabDBESqbsxC7ESQ) (not trained by myself) |0.61904 |45.072% |0.57173 |60.964% |0.52062 |75.789%
|[ArcFace-r50](https://pan.baidu.com/s/1qOIhCauwZNTOCIM9eojPrA)(ms1m-refine-v1 is not trained by myself) |0.61412 |48.155% |0.56749 |63.676% |0.51537 |78.138%
|[ArcFace-r100](https://pan.baidu.com/s/1PeujQbIqFfgARIYAdRt3pw) (not trained by myself) |0.57891 |63.854% |0.53337 |78.129% |0.48079 |89.579%



For more face models, please see [Model-Zoo-for-Face-Recognition](https://github.com/zuoqing1988/ZQCNN-v0.0/wiki/Model-Zoo-for-Face-Recognition)

**Expression recognition**

[FacialEmotion](https://pan.baidu.com/s/1zJtRYv-kSGSCTgpvqc4Iug) Seven types of expressions are trained using Fer2013

**Gender Age Identification**

[GenderAge-ZQ](https://pan.baidu.com/s/1igSpmFt8XBoMk5d4GiXONg) Model trained using [train-GenderAge](https://github.com/zuoqing1988/train-GenderAge)

**Target Detection**

[MobileNetSSD](https://pan.baidu.com/s/1cyly_17cTOJBaCRiiQtWkQ) Format converted from [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

[MobileNetSSD-Mouth](https://pan.baidu.com/s/1_l0Z1R34sOv2R73DB_zXyg) for SampleDetectMouth

**Text detection**

[TextBoxes](https://pan.baidu.com/s/1XOREgRzyimx_AMC9bg8MgQ) Format converted from [TextBoxes](https://github.com/MhLiao/TextBoxes)

**Picture identification**

[NSFW](https://pan.baidu.com/s/1asjZFr3iTliQ4xlNbtKUtw) format transferred from [open_nsfw](https://github.com/yahoo/open_nsfw)

# related articles

(1)[How much accuracy is lost when storing face feature vectors in integers? ](https://zhuanlan.zhihu.com/p/35904005)

(2)[Feature vectors of tens of millions of faces, how to speed up similarity calculation? ](https://zhuanlan.zhihu.com/p/35955061)

(3)[Create a Forward library that is faster than mini-caffe](https://zhuanlan.zhihu.com/p/36410185)

(4)[Accuracy issue of vector dot product](https://zhuanlan.zhihu.com/p/36488847)

(5)[ZQCNN supports Depthwise Convolution and uses mobilenet to modify SphereFaceNet-10](https://zhuanlan.zhihu.com/p/36630082)

(6)[Keep up with the trend of the times and release some source code](https://zhuanlan.zhihu.com/p/37708639)

(7)[ZQCNN supports SSD and is about 30% faster than mini-caffe](https://zhuanlan.zhihu.com/p/40634934)

(8)[ZQCNN’s SSD supports the same model to change the resolution at will](https://zhuanlan.zhihu.com/p/40676503)

(9)[99.78% accuracy face recognition model in ZQCNN format](https://zhuanlan.zhihu.com/p/41197488)

(10)[ZQCNN adds test code for face recognition on the LFW data set](https://zhuanlan.zhihu.com/p/41381883)

(11)[Hold mxnet’s thighs tightly and start writing mxnet2zqcnn](https://zhuanlan.zhihu.com/p/41667828)

(12)[Large-scale face test set and how to create your own face test set](https://zhuanlan.zhihu.com/p/45441865)

(13)[Matrix description of ordinary convolution, mobilenet convolution, and global average pooling](https://zhuanlan.zhihu.com/p/45536594)

(14)[ZQ_FastFaceDetector, a faster and more accurate face detection library](https://zhuanlan.zhihu.com/p/51561288)

**Android compilation instructions**
1. Modify the ndk path and opencv Android sdk path in build.sh
2. Modify CMakeLists.txt
   from the original
   ```
    #add_definitions(-march=native)
    add_definitions(-mfpu=neon)
    add_definitions(-mfloat-abi=hard)
   ```
   Change to
   ```
   #add_definitions(-march=native)
   add_definitions(-mfpu=neon)
   add_definitions(-mfloat-abi=softfp)
   ```
3. This should be able to compile the two libraries ZQ_GEMM and ZQCNN. If you want to compile SampleMTCNN, you can follow the error prompts to modify the parts that cannot be compiled, mainly openmp and timing functions.

