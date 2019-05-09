from math import pow, ceil
from numba import cuda, int32
import numpy as np
import time

N_DIMENSIONS = 0

@cuda.jit("f4(f4[:],f4[:])", device=True)
def rdist(array_a, array_b):
    result = 0.0
    for i in range(array_a.shape[0]):
        tmp = array_a[i] - array_b[i]
        result += tmp * tmp
    
    return result


@cuda.jit("f4(f4[:],f4[:])", device=True)
def rdist_fixed(array_a, array_b):
    result = 0.0
    for i in range(N_DIMENSIONS):
        tmp = array_a[i] - array_b[i]
        result += tmp * tmp
    
    return result

@cuda.jit
def init_arrays(array_a, array_b):
    n_elements = array_a.shape[0]
    
    # elements handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(n_elements / n_threads))
    start_pos = thread_id * thread_size
    end_pos = min((thread_id + 1) * thread_size, n_elements)

    # put some data in arrays
    for i in range(start_pos, end_pos):
        for j in range(N_DIMENSIONS):
            array_a[i][j] = (thread_id * i * j) % (n_elements * N_DIMENSIONS)
            array_b[i][j] = (thread_id * i + j) % (n_elements * N_DIMENSIONS)
    

@cuda.jit
def test_loop_unrolling(
    array_a, array_b, dists, use_dev_fun, use_fixed
):
    n_elements = array_a.shape[0]
    n_loops = 1000
    # elements handled by this thread
    thread_id = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    thread_size = int(ceil(n_elements / n_threads))
    start_pos = thread_id * thread_size
    end_pos = min((thread_id + 1) * thread_size, n_elements)

    # run rdist either with or without device function
    if use_dev_fun:
        if use_fixed:
            for epoch in range(n_loops):
                for i in range(start_pos, end_pos):
                    dists[i] = rdist_fixed(array_a[i], array_b[i])
        else:
            for epoch in range(n_loops):
                for i in range(start_pos, end_pos):
                    dists[i] = rdist(array_a[i], array_b[i])
        
    else:
        for epoch in range(n_loops):
            for i in range(start_pos, end_pos):
                result = 0.0
                for d in range(N_DIMENSIONS):
                    tmp = array_a[i][d] - array_b[i][d]
                    result += tmp * tmp
                dists[i] = result


def test():
    n_threads = 256
    threads_per_block = 32
    n_blocks = n_threads // threads_per_block #dividable values
    
    global N_DIMENSIONS
    N_DIMENSIONS = 2
    n_elements = 100000
    
    d_array_a = cuda.device_array((n_elements, N_DIMENSIONS), dtype=np.float32)
    d_array_b = cuda.device_array((n_elements, N_DIMENSIONS), dtype=np.float32)
    d_dists = cuda.device_array(n_elements, dtype=np.float32)    
    
    # run on gpu
    print("starting kernels with {} threads in {} blocks.".format(n_threads, n_blocks))
    init_arrays[n_blocks, threads_per_block](d_array_a, d_array_b)
    
    # dummy kernel run to get compilation out of time measurement
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists, True, True
    )
    
    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists, True, True
    )
    end=time.time()
    print("kernel using device function with fixed iterator completed in {:.6f}".format(end-start))
    dists_dev_fun_fixed = d_dists.copy_to_host()

    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists, True, False
    )
    end=time.time()
    print("kernel using device function without fixed iterator completed in {:.6f}".format(end-start))
    dists_dev_fun_non_fixed = d_dists.copy_to_host()
    
    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists, False, False
    )
    end=time.time()
    print("kernel without device function completed in {:.6f}".format(end-start))
    dists_non_dev_fun = d_dists.copy_to_host()
    
    # copy result back from device
    
    return (dists_dev_fun_fixed, dists_dev_fun_non_fixed, dists_non_dev_fun)

