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
def test_loop_unrolling(
    array_a, array_b, dists, use_dev_fun
):
    n_elements = array_a.shape[0]
    n_loops = 1000
    
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
    
    # run either rdist with or without device function distances
    if use_dev_fun:
        for epoch in range(n_loops):
            for i in range(start_pos, end_pos):
                dists[i] = rdist_fixed(array_a[i], array_b[i])
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
    n_elements = 10000
    
    array_a = np.zeros((n_elements, N_DIMENSIONS), dtype=np.float32)
    array_b = np.zeros((n_elements, N_DIMENSIONS), dtype=np.float32)
    dists_dev_fun = np.zeros(n_elements, dtype=np.float32)
    dists_non_dev_fun = np.zeros(n_elements, dtype=np.float32)
    
    # copy arrays to device
    d_array_a = cuda.to_device(array_a)
    d_array_b = cuda.to_device(array_b)
    
    d_dists_dev_fun = cuda.to_device(dists_dev_fun)
    d_dists_non_dev_fun = cuda.to_device(dists_non_dev_fun)
    
    # run on gpu
    print("starting kernels with {} threads in {} blocks.", n_threads, n_blocks)
    
    # dummy kernel run to get compilation out of the way
    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists_dev_fun, True
    )
    end=time.time()
    print("dummy run completed in {}", end-start)
    
    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists_dev_fun, True
    )
    end=time.time()
    print("kernel using device function completed in {}", end-start)
    
    start=time.time()
    test_loop_unrolling[n_blocks, threads_per_block](
        d_array_a, d_array_b, d_dists_non_dev_fun, False
    )
    end=time.time()
    print("kernel without device function completed in {}", end-start)
    
    # copy result back from device
    dists_dev_fun = d_dists_dev_fun.copy_to_host()
    dists_non_dev_fun = d_dists_non_dev_fun.copy_to_host()
    
    return dists_dev_fun, dists_non_dev_fun

