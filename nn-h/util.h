#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <assert.h>
using namespace std;

template<int Nbit,unsigned Length>
void StrExt4Str(hls::stream<ap_uint<Nbit> >& in,hls::stream<ap_uint<Nbit> >& out1,hls::stream<ap_uint<Nbit> >& out2,hls::stream<ap_uint<Nbit> >& out3,hls::stream<ap_uint<Nbit> >& out4,unsigned reps = 1){
	for(unsigned i = 0;i < Length*reps;i++){
		ap_uint<Nbit> temp = in.read();
		out1.write(temp);
		out2.write(temp);
		out3.write(temp);
		out4.write(temp);
	}
}

template<int Nbit,unsigned Length>
void StrExt2Str(hls::stream<ap_uint<Nbit> >& in,hls::stream<ap_uint<Nbit> >& out1,hls::stream<ap_uint<Nbit> >& out2,unsigned reps = 1){
	for(unsigned i = 0;i < Length*reps;i++){
		ap_uint<Nbit> temp = in.read();
		out1.write(temp);
		out2.write(temp);
	}
}

//2个stream头接尾得到一个stream
template<int Nbit,unsigned Length>
void Str2LineStr(hls::stream<ap_uint<Nbit> >& out,hls::stream<ap_uint<Nbit> >& in1,hls::stream<ap_uint<Nbit> >& in2){
	for(unsigned i = 0;i < Length;i++){
		ap_uint<Nbit> temp = in1.read();
		out.write(temp);
	}
	for(unsigned i = 0;i < Length;i++){
		ap_uint<Nbit> temp = in2.read();
		out.write(temp);
	}
}

//1个stream中间折断成2个stream
template<int Nbit,unsigned Length>
void StrBreakStr2(hls::stream<ap_uint<Nbit> >& in,hls::stream<ap_uint<Nbit> >& out1,hls::stream<ap_uint<Nbit> >& out2){
	for(unsigned i = 0;i < Length;i++){
		ap_uint<Nbit> temp = in.read();
		out1.write(temp);
	}
	for(unsigned i = 0;i < Length;i++){
		ap_uint<Nbit> temp = in.read();
		out2.write(temp);
	}
}


template<int IStrBit,int OStrBit,unsigned Length>
void ExtendStreamWidth_Length(hls::stream<ap_uint<IStrBit> >& in,hls::stream<ap_uint<OStrBit> >& out,unsigned reps = 1){
	for(int i = 0; i < Length*reps;i++){
#pragma HLS PIPELINE II = 1
		ap_uint<OStrBit> Temp;
		if(OStrBit > IStrBit)
			Temp(OStrBit-1,IStrBit) = 0;
		Temp(IStrBit-1,0)= in.read();
		out.write(Temp);
	}
}


template<int IStrBit,int OStrBit,unsigned Length>
void ReduceStreamWidth_Length(hls::stream<ap_uint<IStrBit> >& in,hls::stream<ap_uint<OStrBit> >& out,unsigned reps = 1){
	for(int i = 0; i < Length*reps;i++){
#pragma HLS PIPELINE II = 1
		ap_uint<IStrBit> InTemp = in.read();
		out.write(InTemp(OStrBit,0));
	}
}

template<int IStrBit,int OStrBit,unsigned Length>		//inLength
void splitStream_Length(hls::stream<ap_uint<IStrBit> >& in,hls::stream<ap_uint<OStrBit> >& out,unsigned reps = 1){
//	static_assert(IStrBit%OStrBit == 0);
	const unsigned times = IStrBit/OStrBit;
	for(int i = 0; i < Length*reps;i++){
#pragma HLS PIPELINE II = times
		ap_uint<IStrBit> InTemp = in.read();
		for(int i = 0;i<times;i++){
			ap_uint<OStrBit> OutTemp = InTemp((i+1)*OStrBit-1, i*OStrBit);
			out.write(OutTemp);
		}
	}
}

template<int IStrBit,int OStrBit,unsigned Length>		//outLength
void mergeStream_Length(hls::stream<ap_uint<IStrBit> >& in,hls::stream<ap_uint<OStrBit> >& out,unsigned reps = 1){
//	static_assert(OStrBit%IStrBit == 0);
	const unsigned times = OStrBit/IStrBit;
	ap_uint<IStrBit> InTemp[times];
#pragma HLS array_partition variable=InTemp complete
	ap_uint<OStrBit> OutTemp;
	for(int i = 0; i < Length*reps;i++){
#pragma HLS PIPELINE II=times
		for(int i = 0;i<times;i++){
			InTemp[i] = in.read();
			OutTemp((i+1)*IStrBit-1,i*IStrBit) = InTemp[i];
		}
		out.write(OutTemp);
	}
}

template<int IOP,int IBit,int OBit,int Length>
void EleExtend(hls::stream<ap_uint<IOP*IBit> >& in,hls::stream<ap_uint<IOP*OBit> >& out,unsigned reps = 1){

	for(int q = 0;q < reps*Length;q++){
		ap_uint<IOP*IBit> Itemp = in.read();
		ap_uint<IOP*OBit> res;
		for(int i = 0;i < IOP;i++){
			res((i+1)*OBit-(OBit+1-IBit),i*OBit) = Itemp((i+1)*IBit-1,i*IBit);
		}
		out.write(res);
	}
}
