import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pycuda.compiler import SourceModule
import numpy as np
from pycuda.gpuarray import vec
from pycuda import gpuarray
import pandas as pd
import time
import sys

drv = cuda
print("%d device(s) found." % drv.Device.count())

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print("Device #%d: %s" % (ordinal, dev.name()))
    print("  Compute Capability: %d.%d" % dev.compute_capability())
    print("  Total Memory: %s KB" % (dev.total_memory()//(1024)))
    atts = [(str(att), value) 
            for att, value in list(dev.get_attributes().items())]
    atts.sort()
    
    for att, value in atts:
        print("  %s: %s" % (att, value))

with open('../kernels/multichannel_shared.cu','r') as fh:
    source_code = fh.read()

# Set up event timers

start = cuda.Event()
end = cuda.Event()


start_free = cuda.mem_get_info()[0]
results = []
for max_registers in [32,64]:
    mod = SourceModule(source_code,options=['--maxrregcount=%d' % max_registers])
    cycfold_multichannel = mod.get_function("cycfold_multichannel")
    for num_channels in [8,16,64]:
        for num_bins in [128,512,2048]:
            for num_lags in [1,16,128,512]:#,1024,2048]:
                for samples_per_bin in [1.07,16,30,256]:
                    for fft_len in [2**15,2**16,2**27]:
                        for overlap in [128,512,2048]:

                            num_bytes=2**27
                            num_pols=2
                            bytes_per_sample = 8 #complex64 (two 32 bit floats)
                            num_samples = num_bytes//(bytes_per_sample*num_pols*num_channels)
                            num_fft = num_samples//fft_len
                            if num_fft == 0:
                                continue
                            print ".",
                            sys.stdout.flush()
                            x = (np.arange(1,num_pols+1)[:,None,None,None]+np.arange(num_channels)[None,:,None,None]*10 
                                 + np.arange(num_fft)[None,None,:,None]*1000 + (np.arange(-overlap//2,fft_len-overlap//2)*1j)[None,None,None,:]).astype('complex64')
                            x[:,:,:,:overlap//2] = np.nan+1j*np.nan
                            x[:,:,:,-overlap//2:] = np.nan+1j*np.nan

                            pol0 = gpuarray.to_gpu(x[0].view(vec.float2).ravel())
                            pol1 = gpuarray.to_gpu(x[1].view(vec.float2).ravel())
                            num_valid = fft_len-overlap
                            phase = np.mod(np.arange(num_fft,dtype='float64')*(num_valid)/(samples_per_bin),num_bins)
                            step = (1./samples_per_bin)*np.ones((num_fft,),dtype='float64')
                            phase_gpu = gpuarray.to_gpu(phase)
                            step_gpu = gpuarray.to_gpu(step)

                            bins_per_fft = (fft_len-overlap)/samples_per_bin
                            xx = np.zeros((num_channels,num_bins,num_lags),dtype='complex64')
                            yy = np.zeros((num_channels,num_bins,num_lags),dtype='complex64')
                            xy = np.zeros((num_channels,num_bins,num_lags),dtype='complex64')
                            hits = np.zeros((num_channels,num_bins,num_lags),dtype=np.uint32)
                            xx2 = gpuarray.to_gpu(xx.view(vec.float2))
                            yy2 = gpuarray.to_gpu(yy.view(vec.float2))
                            xy2 = gpuarray.to_gpu(xy.view(vec.float2))
                            hits_gpu = gpuarray.to_gpu(hits)

                            start.synchronize()

                            start.record()
                            cycfold_multichannel(pol0,pol1,phase_gpu,step_gpu,
                                            np.int32(fft_len), np.int32(overlap), 
                                            np.int32(num_bins), np.int32(num_lags), np.int32(num_fft),
                                            xx2, yy2, xy2, hits_gpu,
                                    block=(min((num_lags, 1024)),1,1), grid=(max((1,num_lags//1024)),num_bins,num_channels))
                            end.record()
                            end.synchronize()
                            elapsed = start.time_till(end)
                            mem_used = start_free-cuda.mem_get_info()[0]
                            #print mem_used/1e6

                            del pol0, pol1, xx2, yy2, xy2, hits_gpu,phase_gpu,step_gpu

                            results.append(dict(elapsed=elapsed,num_lags=num_lags,num_bins=num_bins,
                                                num_fft=num_fft,fft_len=fft_len, overlap=overlap,
                                               samples_per_bin=samples_per_bin,num_channels=num_channels,
                                                mem_used=mem_used,max_registers=max_registers,
                                               num_registers=cycfold_multichannel.num_regs))

results = pd.DataFrame(results)

#print "approx num bins", (num_valid*num_fft/float(samples_per_bin))
#print "approx num turns",(num_valid*num_fft/float(samples_per_bin)/float(num_bins))
#print "approx hits per bin", (num_valid*num_fft)/float(num_bins)h

results['ns_per_sample']=(1e6*results.elapsed/(results.num_channels*results.num_fft*results.fft_len))

filename = time.strftime('%Y-%m-%d_%H%M%S_multichannel_shared_results.h5')
results.to_hdf(filename,'results')