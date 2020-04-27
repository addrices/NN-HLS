#include <ap_int.h>
#include <hls_stream.h>
struct ap_axis{
	ap_uint<128> data;
	ap_int<1> last;
	ap_int<16> keep;
};

template<unsigned StrSize>
void AddLast(
		hls::stream<ap_uint<128> >& in,
		hls::stream<ap_axis>& out,
		unsigned reps = 1
){
	ap_axis temp;
	temp.keep = "0xffff";
	for(unsigned i = 0;i < StrSize*reps-1;i++){
#pragma HLS PIPELINE II = 1
		temp.data = in.read();
		temp.last = 0;
		out.write(temp);
	}
	temp.data = in.read();
	temp.last = 1;
	out.write(temp);
}

template<unsigned StrSize>
void DelHead(
		hls::stream<ap_axis>& in,
		hls::stream<ap_uint<128> >& out,
		unsigned reps = 1
){
	ap_axis temp;
	for(unsigned i = 0;i < StrSize*reps;i++){
#pragma HLS PIPELINE II = 1
		temp = in.read();
		out.write(temp.data);
		if(temp.last == 1)
			break;
	}
}


struct ap_axis_512{
	ap_uint<512> data;
	ap_int<1> last;
	ap_int<64> keep;
};

template<unsigned StrSize>
void AddLast_512(
		hls::stream<ap_uint<512> >& in,
		hls::stream<ap_axis_512>& out,
		unsigned reps = 1
){
	ap_axis_512 temp;
	temp.keep = "0xffffffffffffffff";
	for(unsigned i = 0;i < StrSize*reps-1;i++){
#pragma HLS PIPELINE II = 1
		temp.data = in.read();
		temp.last = 0;
		out.write(temp);
	}
	temp.data = in.read();
	temp.last = 1;
	out.write(temp);
}

template<unsigned StrSize>
void DelHead_512(
		hls::stream<ap_axis_512>& in,
		hls::stream<ap_uint<512> >& out,
		unsigned reps = 1
){
	ap_axis_512 temp;
	for(unsigned i = 0;i < StrSize*reps;i++){
#pragma HLS PIPELINE II = 1
		temp = in.read();
		out.write(temp.data);
		if(temp.last == 1)
			break;
	}
}