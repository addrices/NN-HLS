#define AP_INT_MAX_W 8192
#include "../../nn-h/FcnnLayer.h"
#include "../../nn-h/util.h"
#include "../../nn-h/conv.h"
#include "../../nn-h/pooling.h"
#include "../../nn-h/conv_systolic.h"
#include "config.h"
#include <ap_int.h>
#include <hls_stream.h>



//struct ap_axis{
//	ap_int<512> data;
//	ap_int<1> last;
//	ap_int<64> keep;
//};
//
//template<unsigned StrSize>
//void AddLast(
//		hls::stream<ap_int<512> >& in,
//		hls::stream<ap_axis>& out
//){
//	ap_axis temp;
//	temp.keep = "0xffffffffffffffff";
//	for(unsigned i = 0;i < StrSize;i++){
//#pragma HLS PIPELINE II = 1
//		temp.data = in.read();
//		if(i == StrSize-1)
//			temp.last = 1;
//		else
//			temp.last = 0;
//		out.write(temp);
//	}
//}
//
//template<unsigned StrSize>
//void DelHead(
//		hls::stream<ap_axis>& in,
//		hls::stream<ap_int<512> >& out
//){
//	ap_axis temp;
//	for(unsigned i = 0;i < StrSize;i++){
//#pragma HLS PIPELINE II = 1
//		temp = in.read();
//		out.write(temp.data);
//		if(temp.last == 1)
//			break;
//	}
//}
//
//
//
const ap_int<8*3*3> Weight1[1][16];
const ap_int<8> Bias[32];
//void top(hls::stream<ap_uint<8*1*4> >& in,hls::stream<ap_uint<8*8*4> >& out){
//	const unsigned Batch = 8;
//	ConvLayer_NOPAD_Orbital<Batch,C2_KSIZE,C2_SIZE,C2_INCHANNEL,C2_OUTCHANNEL,1,8,8,1,8,4,12>(in,out,C2_W,C2_B,C2_SCALEBIT,1);
//
//}

//void top(hls::stream<ap_uint<4*20*20> >& act,hls::stream<ap_int<8*20*20> >& wei,hls::stream<ap_int<12*20*20> >& res){
//	//template<unsigned Depth,unsigned ASize,unsigned WSize,unsigned ABit,unsigned WBit,unsigned MBit>
//	//void Orbital_Gemm(ap_uint<ABit*ASize*Depth> activation,ap_int<WBit*WSize*Depth> weight,ap_int<MBit*ASize*WSize> o){
//	for(int i = 0;i < 100;i++){
//		ap_uint<4*20*20>  a = act.read();
//		ap_int<8*20*20>  b = wei.read();
//		ap_int<12*20*20> c;
//		c = Orbital_Gemm<20,20,20,4,8,12>(a,b);
//		res.write(c);
//	}
//}

const ap_int<8*8> Weight2[(16/8)*3*3*(32/8)][8];

void top(hls::stream<ap_uint<16*4> >& in,hls::stream<ap_uint<32*4> >& out){
	C2:ConvLayer_NOPAD_ScaleBit<C2_KSIZE,WBIT,ABIT,C2_MBIT,C2_INCHANNEL,C2_OUTCHANNEL,C2_STRIDE,C2_SIZE,8,8>(in,out,C2_W_,C2_B,C2_SCALEBIT,8);
}



