# NN_HLS
提供了vivado_hls下的CNN实现模块，将CNN网络的实现用nn-h/中对应的函数实现，使用HLS编译生成对应的硬件代码。

## 运行
进入/Dot或者/Gemm目录下
```
$make hls_proj     //在 /Dot/output/下生成HLS项目，进入能够使用gui操作
$make hls_ip       //生成 vivado_hls实现的ip核
$make viv_pro      //生成 vivado项目
```
生成vivado项目后，使用gui打开后，其对应的block design已经将HLS生成的ip核和PS端生成好
。对block设计create wrapper，然后综合实现生成比特流。

## pynq
上传bit流文件，bd设计的tcl文件和/pynq/MNIST_CNN.ipynb至板上，使用jupyter notebook连接运行。