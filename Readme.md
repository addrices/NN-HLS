# NN_HLS
提供了vivado_hls下的CNN实现模块，将CNN网络的实现用nn-h/中对应的函数实现，使用HLS编译生成对应的硬件代码。

提供了MNISTCNN的示例IP。

## 目录说明
```
/Dot  接口是AXI-Stream，提供了PYNQ工程，计算核是向量乘矩阵
/Gemm 接口是AXI-Stream，提供了PYNQ工程，计算核是矩阵乘
/MNISTCNN_axi 接口是axi接口，提供了样例bd工程，可以集成到SOC上去
```

## 运行
进入/Dot或者/Gemm目录下
```
$make hls_proj     //在 /Dot/output/下生成HLS项目，进入能够使用gui操作
$make viv_pro      //生成 vivado项目
```
生成vivado项目后，使用gui打开后，其对应的block design已经将HLS生成的ip核和PS端生成好。对block设计create wrapper，然后综合实现生成比特流。

## HLS IP控制
以MNISTCNN axi为例子：

提供的axilite作为控制接口其中的第一个地址的数据的低四位分别是ap_start,ap_ready,ap_idle,ap_done. 向ap_start写入1即可启动IP的运行，地址偏移0x18中存放着输入数据的地址，需要在启动前写入已经放置好数据的地址。启动后其会将结果放置到地址偏移0x10处。

## 工程的实现说明

```
int top(ap_uint<32> *in){
#pragma HLS INTERFACE m_axi depth=784 port=in bundle=inMem
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW
	const unsigned Batch = 1;
	const unsigned reps = 1;
	ap_uint<32> out[10];
	hls::stream<ap_uint<Batch*C1_INP*ABIT> > C1_in;
	hls::stream<ap_uint<Batch*C1_OUTP*ABIT> > C1_out;
	hls::stream<ap_uint<Batch*C2_OUTP*ABIT> > C2_out;
	hls::stream<ap_uint<Batch*P2_OUTP*ABIT> > P2_out;
	hls::stream<ap_uint<Batch*C3_OUTP*ABIT> > C3_out;
	hls::stream<ap_uint<Batch*P3_OUTP*ABIT> > P3_out;
	hls::stream<ap_uint<Batch*C4_OUTP*ABIT> > C4_out;
	hls::stream<ap_uint<Batch*P4_OUTP*ABIT> > P4_out;

	hls::stream<ap_uint<Batch*F5_OUTPIX*ABIT> > F5_out;
	hls::stream<ap_uint<Batch*F6_OUTPIX*ABIT> > F6_out;

	for(int i = 0;i < 28*28;i++){
		C1_in.write(in[i]);
	}
	C1:ConvLayer_NOPAD_Normal<Batch,C1_KSIZE,C1_SIZE,C1_INCHANNEL,C1_OUTCHANNEL,C1_INP,C1_MIDP_I,C1_MIDP_O,C1_OUTP,C1_STRIDE,WBIT,ABIT,C1_MBIT>(C1_in,C1_out,C1_W,C1_B,C1_SCALEBIT,reps);
	C2:ConvLayer_NOPAD_Normal<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,C2_INP,C2_MIDP_I,C2_MIDP_O,C2_OUTP,C2_STRIDE,WBIT,ABIT,C2_MBIT>(C1_out,C2_out,C2_W,C2_B,C2_SCALEBIT,reps);

	P2:MaxPool_IOP<P2_PSIZE,ABIT,P2_SIZE,P2_CHANNEL*Batch,P2_INP*Batch,P2_OUTP*Batch>(C2_out,P2_out,reps);
	C3:ConvLayer_NOPAD_Normal<Batch,C3_KSIZE,C3_SIZE,C3_INCHANNEL,C3_OUTCHANNEL,C3_INP,C3_MIDP_I,C3_MIDP_O,C3_OUTP,C3_STRIDE,WBIT,ABIT,C3_MBIT>(P2_out,C3_out,C3_W,C3_B,C3_SCALEBIT,reps);
	P3:MaxPool_IOP<P3_PSIZE,ABIT,P3_SIZE,P3_CHANNEL*Batch,P3_INP*Batch,P3_OUTP*Batch>(C3_out,P3_out,reps);
	C4:ConvLayer_NOPAD_Normal<Batch,C4_KSIZE,C4_SIZE,C4_INCHANNEL,C4_OUTCHANNEL,C4_INP,C4_MIDP_I,C4_MIDP_O,C4_OUTP,C4_STRIDE,WBIT,ABIT,C4_MBIT>(P3_out,C4_out,C4_W,C4_B,C4_SCALEBIT,reps);
	P4:MaxPool_IOP<P4_PSIZE,ABIT,P4_SIZE,P4_CHANNEL*Batch,P4_INP*Batch,P4_OUTP*Batch>(C4_out,P4_out,reps);

	F5:FcnnLayer_Batch<Batch,F5_INPIX,F5_OUTPIX,F5_INSIZE,F5_OUTSIZE,WBIT,ABIT,F5_MBIT,F5_DEPTH,F5_WSIZE>(P4_out,F5_out,F5_W,F5_B,F5_SCALEBIT,reps);
	F6:FcnnLayer_Batch<Batch,F6_INPIX,F6_OUTPIX,F6_INSIZE,F6_OUTSIZE,WBIT,ABIT,F6_MBIT,F6_DEPTH,F6_WSIZE>(F5_out,F6_out,F6_W,F6_B,F6_SCALEBIT,reps);
	for(int i = 0;i < 10;i++){
		out[i] = F6_out.read();
	}
	int res = (int)max_array<10>(out);
	return res;
}
```
加速器中的数据以流的形式进入，并一层一层的流过HLS实现的加速模块，nn-h文件夹中提供了CNN的基本组件，只需要将正确size的数据流传入各个函数中便能够得到结果。```#pragma HLS DATAFLOW```指示HLS将其中的各个模块设计成流水线的结构，层之间的HLS::stream将会被设计成FIFO在各个层之间连接。

### 各个函数的说明情况

#### conv
```
template<unsigned Batch,
        unsigned KSize,
        unsigned Size,
        unsigned InChannel,
        unsigned OutChannel,
        unsigned InP,
        unsigned MidP_i,
        unsigned MidP_o,
        unsigned OutP,
        unsigned Stride,
        unsigned WBit,
        unsigned ABit,
        unsigned MBit>
void ConvLayer_NOPAD_Normal(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*MidP_o*MidP_i> Weight[KSize*KSize][OutChannel/MidP_o][InChannel/MidP_i],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1)

void ConvLayer_NOPAD_Normal_Gemm(hls::stream<ap_uint<Batch*InP*ABit> >& in,hls::stream<ap_uint<Batch*OutP*ABit> >& out,const ap_int<WBit*KSize*KSize> Weight[OutChannel][InChannel],const ap_int<WBit> Bias[OutChannel],const unsigned Scale,unsigned reps = 1){
```
这俩函数编写了Conv2d层的计算,其中ConvLayer_NOPAD_Normal的核心计算核是向量乘矩阵，ConvLayer_NOPAD_Normal_Gemm的核心计算核是矩阵乘。

template参数说明

| 名字       | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| Batch      | Batch的大小                                                  |
| KSize      | 卷积核大小                                                   |
| Size       | Feature Map的大小                                            |
| InChannel  | 输入Channel                                                  |
| OutChannel | 输出Channel                                                  |
| InP        | 输入stream中包含多少个点（沿InChannel方向）,被InChannel整除  |
| MidP_i     | 指示中间计算核的大小，被InP整除                              |
| MidP_o     | 指示中间计算核的大小，被OutP整除                             |
| OutP       | 输出stream中包含多少个点（沿OutChannel方向）,被OutChannel整除 |
| Stride     | 步长                                                         |
| WBit       | 权重的位数                                                   |
| ABit       | 激活值的位数                                                 |
| MBit       | 指示存放中间结果的位数                                       |

建议使用Dot版本的，Gemm好像有点问题综合出来的。

Dot的乘法器数量为Batch\*MidP_i\*MidP_o。

Gemm的乘法器数量为Batch\*KSize\*Ksize\*MidP_o。

#### Pooling

```
template<unsigned WinSize,
unsigned IOBit,
unsigned Size,
unsigned Channel,
unsigned InP,
unsigned OutP>
void MaxPool_IOP(hls::stream<ap_uint<IOBit*InP> >& in, hls::stream<ap_uint<IOBit*OutP> >& out,unsigned reps = 1)
```

template说明

| 名字    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| WinSize | Pooling窗口大小                                              |
| IOBit   | 使用的bit位数                                                |
| Size    | FeatureMap大小                                               |
| Channel | Channel大小                                                  |
| InP     | 指示进入的Stream的Size                                       |
| OutP    | 指示出去的Stream的Size //一般设置成和InP一样大，这样可以保证中间流水线的畅通 |

这里的Batch可以合并到Channel，InP，OutP中去，达到Batch的效果。

#### FcnnLayer_Batch

```
template<unsigned Batch,
        unsigned InPix,
		unsigned OutPix,
		unsigned InSize,
		unsigned OutSize,
		unsigned WBit,
		unsigned ABit,
        unsigned MBit,
        unsigned Depth,
        unsigned WSize>	//has InPix and OutPix
void FcnnLayer_Batch(hls::stream<ap_uint<Batch*ABit*InPix> >& in,hls::stream<ap_uint<Batch*ABit*OutPix> >& out,const ap_int<WBit> Weight[OutSize][InSize],const ap_int<WBit> Bias[OutSize],const unsigned ScaleBit,unsigned reps = 1)
```

template说明

| 名字    | 说明                   |
| ------- | ---------------------- |
| Batch   | Pooling窗口大小        |
| InPix   | 全连接层输入的点的个数 |
| OutPix  | 全连接层输出的点的个数 |
| InSize  | 指示进入的Stream的Size |
| OutSize | 指示出来的Stream的Size |
| WBit    | 权重的Bit数            |
| ABit    | 激活值的Bit数          |
| MBit    | 计算的中间结果的Bit数  |
| Depth   | 计算核矩阵乘的深度     |

计算核大小为Batch*Depth



## pynq

PYNQ的deploy文件好像坏了，手上暂时没有pynq板子，先跑不了。