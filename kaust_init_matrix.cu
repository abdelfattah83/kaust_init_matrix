#include<stdio.h>
#include<vector>
#include<cuda_runtime.h>

#define THREAD_X (64)
#define THREAD_Y (2)

#define NITER    (10)

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ __inline__ 
void init_matrix_device(int m, int n, T* A, int ldda, int gtx, int gty) 
{
	if(gtx < m && gty < n) {
	    T value = (T)gtx + (T)gty * (T)ldda;
		A[gty * ldda + gtx] = value;
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ 
void init_matrix_kernel_batched(int m, int n, T** dA_array, int ldda)
{
	const int batchid = blockIdx.z;
	const int gtx     = blockIdx.x * blockDim.x;
	const int gty     = blockIdx.y * blockDim.y;

    init_matrix_device(m, n, dA_array[batchid], ldda, gtx, gty);
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void init_matrix_batched(int m, int n, T** dA_array, int ldda, int batch_size)
{
	dim3 threads(THREAD_X, THREAD_Y, 1);

	const int gridx = (m + (THREAD_X-1)) / THREAD_X;
	const int gridy = (n + (THREAD_Y-1)) / THREAD_Y;
	// grid-z dimension has a max value of ~65k
	// for batches larger than this, we should use a loop
	const int max_batch = 60000;
	for(int i = 0; i < batch_size; i+=max_batch) {
		int gridz = min(max_batch, batch_size-i);
		dim3 grid(gridx, gridy, gridz);
		init_matrix_kernel_batched<<<grid, threads, 0, 0>>>(m, n, dA_array, ldda);
	}
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	size_t batch_size = 1000;
	size_t m          = 50;
	size_t n          = m;

    if(argc > 1)
		batch_size = (size_t)atoi( argv[1] );
	
	if(argc > 2) {
		m = (size_t)atoi(argv[2]);
		n = m;
	}

	if(argc > 3)
		n = (size_t)atoi(argv[3]);
	
	// check for quick return
	if(batch_size <= 0 || m <= 0 || n <= 0) {
	    printf("One of the dimensions or batch_size is <= 0, exiting\n");
		return 0;
	}

    printf("Initializing a batch of %d matrices -- size %d x %d\n", batch_size, m, n);
    const size_t ldda = m;

	// alloc matrices on GPU
	size_t sizeA = batch_size * ldda * n * sizeof(double);
	printf("Data size = %.2f GB\n", sizeA / (1024. * 1024. * 1024.));
    double* dA = NULL;
	cudaError_t e = cudaMalloc((void**)&dA, sizeA);
	if(e != cudaSuccess) {
	    printf("Error allocating dA: %s\n", cudaGetErrorString(e));
	}

	// setup pointer array on cpu and copy to gpu
	std::vector<double*>hA_array(batch_size);
	hA_array[0] = dA;
	for(int i = 1; i < hA_array.size(); i++) {
	    hA_array[i] = hA_array[i-1] + ldda * n;
	}

    double** dA_array = NULL;
	e = cudaMalloc((void**)&dA_array, batch_size * sizeof(double*));
    if(e != cudaSuccess) {
        printf("Error allocating dA_array: %s\n", cudaGetErrorString(e));
    }
	cudaMemcpy(dA_array, hA_array.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);


    // for timing
    cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop,  0);
	

    // warmup run
	init_matrix_batched(m, n, dA_array, ldda, batch_size);
	
	// launch kernel
	cudaEventRecord(start);
	for(int i = 0; i < NITER; i++) {
		init_matrix_batched(m, n, dA_array, ldda, batch_size);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	// print time
	float ms = 0.;
	cudaEventElapsedTime(&ms,start,stop);
	ms /= (float)NITER;
	float bw = (sizeA / 1e9) / (ms / 1000.);
	printf("Initialization time: %.2f ms -- %.2f GB/s\n", ms, bw);


    if(dA       != NULL) cudaFree(dA);
    if(dA_array != NULL) cudaFree(dA_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	return 0;
}

