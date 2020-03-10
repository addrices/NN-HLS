#pragma once
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include "conv_systolic.h"



/*                                                               WEIGHT
 *                                                |--------------OutSize-------------|
 *                                        ------  ++++++++++++++++++++++++++++++++++++
 *                                           |    +       +        +        +        +
 *                                           |    +       +        +        +        +
 *                                           |    +       +        +        +        +
 *                                           |    +       +        +        +        +
 *                                           |    +       +        +        +        +
 *                                        InSize  +       +        +        +        +
 *                                           |    +       +        +        +        +
 *                                           |    ++++++++++++++++++++++++++++++++++++ ---
 *                                           |    +       +        +        +        +  |
 *                                           |    +       +        +        +        + Depth
 *                                        ------  ++++++++++++++++++++++++++++++++++++ ---
 *           |--------------InSize------------|   |-WSize-|
 *   -----   +++++++++++++++++++++++++++++++++
 *     |     +                      +        +
 *   Batch   +     ACT              +        + 
 *     |     +                      +        + 
 *   -----   +++++++++++++++++++++++++++++++++
 *                                  |-Depth--|
 * 
 */
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
void FcnnLayer_Batch(hls::stream<ap_uint<Batch*ABit*InPix> >& in,hls::stream<ap_uint<Batch*ABit*OutPix> >& out,const ap_int<WBit> Weight[OutSize][InSize],const ap_int<WBit> Bias[OutSize],const unsigned ScaleBit,unsigned reps = 1){
    assert(InSize%Depth == 0);
    assert(OutSize%OutSize == 0);
    assert(Depth%InPix == 0);
    assert(InSize%WSize == 0);
    const ap_uint<ABit+1> limit = (1 << ABit);
	ap_int<MBit> result[OutSize][Batch];
    const unsigned InPack1 = InSize/Depth;
    const unsigned InPack2 = Depth/InPix;
    const unsigned WPack = OutSize/WSize;
	const unsigned Pack = OutSize/OutPix;
	for(unsigned rep = 0; rep < reps;rep++){
        for(unsigned q = 0; q < OutSize;q++){
            for(unsigned b = 0;b < Batch;b++){
                result[q][b] = Bias[q];
            }
        }

        ap_uint<Batch*ABit*Depth> act;
        ap_uint<WSize*WBit*Depth> wei;
		FcPipe:for(unsigned i = 0;i < InPack1;i++){
            //cout << "act" << endl;
            for(unsigned j = 0;j < InPack2;j++){
                ap_uint<Batch*ABit*InPix> Rin = in.read();
                for(unsigned a = 0;a < InPix;a++){
                    for(unsigned b = 0;b < Batch;b++){
                        unsigned OFF = j*InPix + b*Depth + a;
                        unsigned ROFF = a*Batch + b;
                        act((OFF+1)*ABit-1,OFF*ABit) = Rin((ROFF+1)*ABit-1,ROFF*ABit);
                        //cout << OFF << " " << act((OFF+1)*ABit-1,OFF*ABit)  << " ";
                    }
                    //cout << endl;
                }
            }
            //cout << endl;
            for(unsigned wpack = 0;wpack < WPack;wpack++){
                //cout << "Weight" << endl;
                for(unsigned a = 0;a < WSize;a++){          //OutSize
                    unsigned OutS = wpack * WSize + a;
                    for(unsigned b = 0;b < Depth;b++){      //InSize
                        unsigned InS = i * Depth + b;
                        unsigned Offset = a * Depth+b;
                        wei((Offset+1)*WBit-1,Offset*WBit) = Weight[OutS][InS];
                        //cout << Weight[OutS][InS] << " ";
                    }
                    //cout << endl;
                }
                //cout << endl;
                ap_int<MBit*Batch*WSize> res = Orbital_Gemm<Depth,Batch,WSize,ABit,WBit,MBit>(act,wei);
                //cout << endl;
                for(unsigned m = 0;m < WSize;m++){
                    unsigned OutS = wpack * WSize + m;
                    for(unsigned b = 0;b < Batch;b++){
                        unsigned offset = m*Batch+b;
                        //cout << "OutS" << OutS << " " << result[OutS][b] << " ";
                        result[OutS][b] += res((offset+1)*MBit-1,(offset)*MBit);
                        //cout << " " << result[OutS][b] << " ";
                    }
                    //cout << endl;
                }
            }
		}
		//clamp(0,1<<ABit)
        //cout << "result" << endl;
		for(unsigned a = 0; a < OutSize;a++){
            for(unsigned b = 0; b < Batch;b++){
                //cout << a << "  " << result[a][b];
                const unsigned HALF = (1 << (ScaleBit-1));
                if(result[a][b] < 0)
                    result[a][b] = 0;
                result[a][b] = (result[a][b]+HALF) >> ScaleBit;
                if(result[a][b] >= limit)
                    result[a][b] = limit - 1;
                //cout << " out " << result[a][b] << " ";
            }
            //cout << endl;
		}
		ap_uint<Batch*ABit*OutPix> OutTemp;
		for(unsigned w = 0;w < Pack;w++){
			for(unsigned e = 0;e < OutPix;e++){
#pragma HLS UNROLL
                for(unsigned b = 0;b < Batch;b++){
#pragma HLS UNROLL
                    unsigned offset = e*Batch+b;
                    unsigned OutS = w*OutPix + e;
				    ap_uint<ABit> OP = result[OutS][b];
				    OutTemp((offset+1)*ABit-1,offset*ABit) = OP;
                }
			}
			out.write(OutTemp);
            //cout << "write" << w << " " << OutTemp << endl;
		}
	}
	return;
}
