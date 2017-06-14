#include <stdio.h>

__global__ void cycfold_multichannel(const float2 *pol0, const float2 *pol1,
        const double *phase, const double *step,
        const int fftlen, const int overlap, const int nbin, const int nlag,
        const int num_fft,
        float2 *xx, float2 *yy, float2 *xy, unsigned *hits) {

	// Lag is specified with threadIdx.x and blockIdx.x since there could be
	// more lags than allowed threads.
    const int ilaga = threadIdx.x;
    const int nlaga = blockDim.x;
    const int ilagb = blockIdx.x;
    const int ilag = ilagb*nlaga + ilaga;
    // Phase bin is blockIdx.y
    const int ibin = blockIdx.y;
    // Filterbank channel is blockIdx.z
    const int ichan = blockIdx.z;


    const int num_valid_samples = fftlen - overlap;

	// accumulators for the various lag terms
    float2 foldxxlag = make_float2(0,0);
    float2 foldyylag = make_float2(0,0);
    float2 foldxylag = make_float2(0,0);
    // Number of hits for this phase/lag bin
    int foldcount = 0;


    for (int ifft=0; ifft < num_fft; ifft++){
    	//Pointers to the first valid sample for this channel and fft
        const float2 *ptr0 = pol0 +ichan*fftlen*num_fft + ifft*fftlen + overlap/2;
        const float2 *ptr1 = pol1 + ichan*fftlen*num_fft + ifft*fftlen + overlap/2;
        // Fold info
        const double bin0 = phase[ifft];
        const double bins_per_sample = step[ifft];   // bins/sample
        const double samples_per_bin = 1.0/bins_per_sample; // samples/bin
        const int num_turns = ((double)num_valid_samples*bins_per_sample)/(double)nbin + 2;

        // Loop over number of pulse periods in data block
        for (int iturn=0; iturn<num_turns; iturn++) {

            // Determine range of samples needed for this bin, turn
            int samp0 = samples_per_bin*((double)ibin-bin0+(double)iturn*nbin)+0.5;
            int samp1 = samples_per_bin*((double)ibin-bin0+(double)iturn*nbin+1)+0.5;

            // Range checks
            if (samp0<0) { samp0=0; }
            if (samp1<0) { samp1=0; }
            if (samp0>num_valid_samples) { samp0=num_valid_samples; }
            if (samp1>num_valid_samples) { samp1=num_valid_samples; }

            // Read in and add samples
            int lag_index;
            for (int isamp=samp0; isamp<samp1; isamp++) {
                lag_index = isamp + ilag -nlag/2;
                if((lag_index >= 0) && (lag_index < num_valid_samples)){
                    float2 p0 = ptr0[isamp];
                    float2 p0lag = ptr0[lag_index];
                    float2 p1 = ptr1[isamp];
                    float2 p1lag = ptr1[lag_index];
                    // <Pol0 x Pol0_lag*>
                    foldxxlag.x += p0.x*p0lag.x + p0.y*p0lag.y;
                    foldxxlag.y += p0.y*p0lag.x - p0.x*p0lag.y;

                    // <Pol1 x Pol1_lag*>
                    foldyylag.x += p1.x*p1lag.x + p1.y*p1lag.y;
                    foldyylag.y += p1.y*p1lag.x - p1.x*p1lag.y;

                    // <Pol0 x Pol1_lag*>
                    foldxylag.x += p0.x*p1lag.x + p0.y*p1lag.y;
                    foldxylag.y += p0.y*p1lag.x - p0.x*p1lag.y;
                    foldcount++;
                }
            }
        }

    }
    xx[ichan*nlag*nbin+nlag*ibin+ilag] = foldxxlag;
    yy[ichan*nlag*nbin+nlag*ibin+ilag] = foldyylag;
    xy[ichan*nlag*nbin+nlag*ibin+ilag] = foldxylag;
    hits[ichan*nlag*nbin+nlag*ibin+ilag] = foldcount;
}
