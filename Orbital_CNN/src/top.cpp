#include "nn-h/conv.h"
#include "nn-h/pooling.h"
#include "nn-h/util.h"
#include "nn-h/activate.h"
#include "nn-h/FcnnLayer.h"
#include <ap_int.h>
#include <hls_stream.h>

struct ap_axis{
	ap_int<512> data;
	ap_int<1> last;
	ap_int<64> keep;
};

template<unsigned StrSize>
void AddLast(
		hls::stream<ap_int<512> >& in,
		hls::stream<ap_axis>& out
){
	ap_axis temp;
	temp.keep = "0xffffffffffffffff";
	for(unsigned i = 0;i < StrSize;i++){
#pragma HLS PIPELINE II = 1
		temp.data = in.read();
		if(i == StrSize-1)
			temp.last = 1;
		else
			temp.last = 0;
		out.write(temp);
	}
}

template<unsigned StrSize>
void DelHead(
		hls::stream<ap_axis>& in,
		hls::stream<ap_int<512> >& out
){
	ap_axis temp;
	for(unsigned i = 0;i < StrSize;i++){
#pragma HLS PIPELINE II = 1
		temp = in.read();
		out.write(temp.data);
		if(temp.last == 1)
			break;
	}
}



// Dot
/*
void top(ap_int<IBit1*Ksize1*Ksize1*InChannel1>& in,ap_int<MBit1*OutChannel1>& out){
#pragma HLS INTERFACE ap_ovld both port=in
#pragma HLS INTERFACE ap_ovld both port=out
#pragma HLS DATAFLOW
	const int P = InChannel1*Ksize1*Ksize1;
	const int InBlockSize = IBit1*Ksize1*Ksize1*InChannel1;
	ap_int<KBit1*P> DotWeight[OutChannel1] = {2,4,13,23,45,12,3,4,5,23,15,16,23,32,25,14};
	ap_int<MBit1> OutSinglePix[OutChannel1];
	ap_int<IBit1*Ksize1*Ksize1*InChannel1> in1[OutChannel1];
	for(int l = 0; l < OutChannel1;l++){
#pragma HLS UNROLL
		in1[l] = in;
	}
	for(int l = 0; l < OutChannel1;l++){
#pragma HLS UNROLL
		OutSinglePix[l] = Dot<InChannel1*Ksize1*Ksize1,IBit1,KBit1,MBit1>(DotWeight[l],in1);
	}

	for(int q = 0; q < OutChannel1;q++){
#pragma HLS UNROLL
		out((q+1)*MBit1-1,q*MBit1) = OutSinglePix[q];
	}
}*/



//#define GEMMSIZE 3

//void aTop(hls::stream<ap_int<8> > &in,hls::stream<ap_int<8> > &out);
//using namespace hls;

/*
void Gemm_(ap_int<8> weight[GEMMSIZE][GEMMSIZE],ap_int<8> activation[GEMMSIZE][GEMMSIZE],ap_int<8> o[GEMMSIZE][GEMMSIZE]){
	ap_int<8> W[GEMMSIZE][GEMMSIZE],A[GEMMSIZE][GEMMSIZE],O[GEMMSIZE][GEMMSIZE];
	for(int a = 0; a < GEMMSIZE; a++){
#pragma HLS UNROLL
		for(int b = 0; b < GEMMSIZE; b++){
#pragma HLS UNROLL
			O[a][b] = 0;
			A[a][b] = activation[(a+b)%GEMMSIZE][b];
			W[a][b] = weight[a][(a+b)%GEMMSIZE];
		}
	}
	for(int i = 0;i < GEMMSIZE;i++){
#pragma HLS PIPELINE II = 1
		for(int j = 0; j < GEMMSIZE; j++){
#pragma HLS UNROLL
			for(int k = 0; k < GEMMSIZE;k++){
#pragma HLS UNROLL
				O[j][k] = O[j][k] + W[j][k] * A[j][k];
				W[j][k] = W[j][(k+1)%GEMMSIZE];
				A[j][k] = A[(j+1)%GEMMSIZE][j];
			}
		}
	}
	for(int c = 0; c < GEMMSIZE; c++){
#pragma HLS UNROLL
		for(int d = 0; d < GEMMSIZE; d++){
#pragma HLS UNROLL
			o[c][d] = O[c][d];
		}
	}
	return;
}
*/


//ap_int<8> k1_8[9] = {1,2,3,4,5,6,7,8,9};
//ap_int<8> k2[9] = {3,2,1,6,5,4,9,8,7};
ap_int<32> k1_32[9] = {1,2,3,4,5,6,7,8,9};
ap_int<32> k2_32[9] = {1,2,3,4,5,1,7,8,9};
ap_int<32> k3_32[9] = {1,2,3,4,5,4,7,8,9};
ap_int<32> k4_32[9] = {1,2,3,4,5,4,7,8,9};
double k1_d[9] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.0};
double k2_d[9] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.0};
double k3_d[9] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.0};
double k4_d[9] = {1.2,2.3,3.4,4.5,5.6,6.7,7.8,8.9,9.0};
/*
void four2one(stream<ap_int<8> >& in1,stream<ap_int<8> >& in2,stream<ap_int<8> >& in3,stream<ap_int<8> >& in4,stream<ap_int<32> >& out){
	for(int j = 0;j < 48*64;j++){
		ap_int<8> m1 = in1.read();
		ap_int<8> m2 = in2.read();
		ap_int<8> m3 = in3.read();
		ap_int<8> m4 = in4.read();
		ap_int<32> outm = m1+(m2<<8)+(m3<<16)+(m4<<24);
		out.write(outm);
	}
}

void one2four(stream<ap_int<8> >& out1,stream<ap_int<8> >& out2,stream<ap_int<8> >& out3,stream<ap_int<8> >& out4,stream<ap_int<32> >& in){
	for(int i = 0;i < 48*64;i++){
		ap_int<32> num = in.read();
		ap_int<8> n1 = num%256;
		ap_int<8> n2 = (num>>8)%256;
		ap_int<8> n3 = (num>>16)%256;
		ap_int<8> n4 = (num>>24)%256;
		out1.write(n1);
		out2.write(n2);
		out3.write(n3);){
		out4.write(n4);
	}
}*/

/*
void aTop(stream<ap_int<32> > &in,stream<ap_int<32> > &out){
#pragma HLS DATAFLOW
	stream<ap_int<8> > in1("in1");
	stream<ap_int<8> > in2("in2");
	stream<ap_int<8> > in3("in3");
	stream<ap_int<8> > in4("in4");
	stream<ap_int<8> > out1("out1");
	stream<ap_int<8> > out2("out2");
	stream<ap_int<8> > out3("out3");
	stream<ap_int<8> > out4("out4");

	one2four(in1,in2,in3,in4,in);
	Conv_Top<3,1,8,8,8,8,8,48,64> (in1,out1,k1);
	Conv_Top<3,1,8,8,8,8,8,48,64> (in2,out2,k1);
	Conv_Top<3,1,8,8,8,8,8,48,64> (in3,out3,k1);
	Conv_Top<3,1,8,8,8,8,8,48,64> (in4,out4,k1);
	four2one(out1,out2,out3,out4,out);

	return;
}
*/

/*
void top(hls::stream<double >& in,hls::stream<double > &out){
#pragma HLS DATAFLOW

	hls::stream<ap_int<32> > temp1,temp2,temp3,temp4;
	hls::stream<ap_int<32> > temp5,temp7,temp6,temp8;
	hls::stream<ap_int<32> > tempo1,tempo2,tempo3,tempo4;
	StrExt4Str<32,3072>(in,temp5,temp6,temp7,temp8);
	Conv_N<3,1,32,32,32,32,48,64> (temp5,temp1,k1_32);
	Conv_N<3,1,32,32,32,32,48,64> (temp6,temp2,k2_32);
	Conv_N<3,1,32,32,32,32,48,64> (temp7,temp3,k3_32);
	Conv_N<3,1,32,32,32,32,48,64> (temp8,temp4,k4_32);
	MaxPooling_Str<2,32,46,62>(temp1,tempo1);
	MaxPooling_Str<2,32,46,62>(temp2,tempo2);
	MaxPooling_Str<2,32,46,62>(temp3,tempo3);		   	   	   	   	   	   	   	   	   	   	   	   	   	   	   {{1,2,3,4,5,6,7,8,9},{1,2,3,4,5,6,7,8,9},{1,2,3,4,5,6,7,8,9}},\

	MaxPooling_Str<2,32,46,62and MobileNet, with the right mix
of 1-, 2- and 3-bit parameters that average to just 1.4 bits can equal the accuracy
of homogeneous 2-bit versions of these networks. Further, we provide analyses
to show that the heterogeneously binarized systems yield FPGA- and ASIC-based
implementations that are correspondingly more efficient i>(temp4,tempo4);
	for(int i = 0;i < 46;i++){
		for(int j = 0;j < 62;j++){
			ap_int<32> T1 = tempo1.read();
			ap_int<32> T2 = tempo2.read();
			ap_int<32> T3 = tempo3.read();
			ap_int<32> T4 = tempo4.read();
			ap_int<32> T = T1+T2+T3+T4;
			out.write(T);
		}
	}
}*/

/*
void top(hls::stream<ap_int<32> >& in,hls::stream<ap_int<32> >& out){
#pragma HLS DATAFLOW
	//hls::stream<ap_int<32> > temp1;
	Conv_N<3,1,32,32,32,32,48,64> (in,out,k1_32);
	//MaxPooling_Str<2,32,46,62>(temp1,out);

}
*/


/*
void Gemm(ap_int<8> &w1,ap_int<8> &w2,ap_int<8> &w3,ap_int<8> &w4,ap_int<8> &w5,ap_int<8> &w6,ap_int<8> &w7,ap_int<8> &w8,ap_int<8> &w9,\
		ap_int<8> &a1,ap_int<8> &a2,ap_int<8> &a3,ap_int<8> &a4,ap_int<8> &a5,ap_int<8> &a6,ap_int<8> &a7,ap_int<8> &a8,ap_int<8> &a9,\
		ap_int<8> &o1,ap_int<8> &o2,ap_int<8> &o3,ap_int<8> &o4,ap_int<8> &o5,ap_int<8> &o6,ap_int<8> &o7,ap_int<8> &o8,ap_int<8> &o9){
#pragma HLS INTERFACE ap_ovld port=w1
#pragma HLS INTERFACE ap_ovld port=a1
#pragma HLS INTERFACE ap_ovld port=o1
#pragma HLS INTERFACE ap_ovld port=w2
#pragma HLS INTERFACE ap_ovld port=a2
#pragma HLS INTERFACE ap_ovld port=o2
#pragma HLS INTERFACE ap_ovld port=w3
#pragma HLS INTERFACE ap_ovld port=a3
#pragma HLS INTERFACE ap_ovld port=o3
#pragma HLS INTERFACE ap_ovld port=w4
#pragma HLS INTERFACE ap_ovld port=a4
#pragma HLS INTERFACE ap_ovld port=o4
#pragma HLS INTERFACE ap_ovld port=w5
#pragma HLS INTERFACE ap_ovld port=a5
#pragma HLS INTERFACE ap_ovld port=o5
#pragma HLS INTERFACE ap_ovld port=w6
#pragma HLS INTERFACE ap_ovld port=a6
#pragma HLS INTERFACE ap_ovld port=o6
#pragma HLS INTERFACE ap_ovld port=w7
#pragma HLS INTERFACE ap_ovld port=a7
#pragma HLS INTERFACE ap_ovld port=o7
#pragma HLS INTERFACE ap_ovld port=w8
#pragma HLS INTERFACE ap_ovld port=a8
#pragma HLS INTERFACE ap_ovld port=o8
#pragma HLS INTERFACE ap_ovld port=w9
#pragma HLS INTERFACE ap_ovld port=a9
#pragma HLS INTERFACE ap_ovld port=o9
	ap_int<8> weight[GEMMSIZE][GEMMSIZE],activation[GEMMSIZE][GEMMSIZE],out[GEMMSIZE][GEMMSIZE];
	weight[0][0] = w1; activation[0][0] = a1;
	weight[0][1] = w2; activation[0][1] = a2;
	weight[0][2] = w3; activation[0][2] = a3;
	weight[1][0] = w4; activation[1][0] = a4;
	weight[1][1] = w5; activation[1][1] = a5;
	weight[1][2] = w6; activation[1][2] = a6;
	weight[2][0] = w7; activation[2][0] = a7;
	weight[2][1] = w8; activation[2][1] = a8;
	weight[2][2] = w9; activation[2][2] = a9;
	Gemm_(weight,activation,out);
	o1 = out[0][0];o2 = out[0][1]; o3 = out[0][2];
	o4 = out[1][0];o5 = out[1][1]; o6 = out[1][2];
	o7 = out[2][0];o8 = out[2][1]; o9 = out[2][2];
	return;
}

void conv_top(int *img,int *ker,int *ret){
//#pragma HLS INTERFACE s_axilite port=h bundle=control
//#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=img bundle=hostmem depth=256
#pragma HLS INTERFACE s_axilite port=img bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=ker bundle=hostmem depth=9*9
#pragma HLS INTERFACE s_axilite port=ker bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=ret bundle=hostmem depth=256*9
#pragma HLS INTERFACE s_axilite port=ret bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


	static int buffer[16][16];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
	static int conv_kernel[9][9];			//batch = 9
#pragma HLS ARRAY_PARTITION variable=conv_kernel complete dim=0

	memcpy(buffer, img, 16*16 * sizeof(int));
	memcpy(conv_kernel, img, 9*9 * sizeof(int));

	int conv;
	int R[9][16][16];

	int GEMM_A[9][9];
	int OUT[9][9];

	for(int i = 0; i < 16; i++){
		for(int j = 0; j < 16; j+=8){
			for(int k = 0; k < 9;k++){
				for(int m = 0; m < 3;m++){
					for(int n = 0; n < 3;n++){
						if(i+m-1<0 || i+m-1>15||j+k+n-1<0||j+k+n-1>15||k==8)
							GEMM_A[3*m+n][k] = 0;
						else
							GEMM_A[3*m+n][k] = buffer[i+m-1][j+k+n-1];
					}
				}
			}
			Gemm(conv_kernel,GEMM_A,OUT);
			for(int o1 = 0; o1 < 9; o1++){
				for(int o2 = 0; o2 < 8; o2++){
					R[o1][i][j+o2] = OUT[o1][o2];
				}
			}
		}
	}
	for(int e=0;e < 9;e++){
		for(int r=0;r < 16;r++){
			for(int t=0;t < 9;t++){
				ret[e*16*16+r*16+t] = R[e][r][t];
			}
		}
	}
	return;
}
*/
