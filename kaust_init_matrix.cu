#include<stdio.h>
#include<vector>
#include<cuda_runtime.h>

#define THREAD_X (64)
#define THREAD_Y (2)

#define NITER    (10)

/*********************************************************************************/
/* device function to initialize, shared between uniform and non-uniform batches */
/*********************************************************************************/
template<typename T>
__device__ __inline__
void init_matrix_device(int m, int n, T* A, int ldda, int gtx, int gty)
{
	if(gtx < m && gty < n) {
	    T value = (T)gtx + (T)gty * (T)ldda;
		A[gty * ldda + gtx] = value;
	}
}

/******************************************************************************/
/*                 CODES FOR UNIFORM BATCH GENERATION                         */
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__
void init_matrix_kernel_batched(int m, int n, T** dA_array, int ldda)
{
	const int batchid = blockIdx.z;
	const int gtx     = blockIdx.x * blockDim.x + threadIdx.x;
	const int gty     = blockIdx.y * blockDim.y + threadIdx.y;

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
		init_matrix_kernel_batched<<<grid, threads, 0, 0>>>(m, n, dA_array+i, ldda);
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void init_matrix_batched_test(int m, int n, int batch_size)
{
    printf("Uniform Batch Generation Test\n");
    printf("=============================\n");
    printf("Initializing a uniform batch of %d matrices -- size %d x %d\n", batch_size, m, n);
    const size_t ldda = m;

	// alloc matrices on GPU
	size_t sizeA = batch_size * ldda * n * sizeof(T);
	printf("Data size = %.2f GB\n", sizeA / (1024. * 1024. * 1024.));
    T* dA = NULL;
	cudaError_t e = cudaMalloc((void**)&dA, sizeA);
	if(e != cudaSuccess) {
	    printf("Error allocating dA: %s\n", cudaGetErrorString(e));
	}

	// setup pointer array on cpu and copy to gpu
	std::vector<T*>hA_array(batch_size);
	hA_array[0] = dA;
	for(int i = 1; i < hA_array.size(); i++) {
	    hA_array[i] = hA_array[i-1] + ldda * n;
	}

    T** dA_array = NULL;
	e = cudaMalloc((void**)&dA_array, batch_size * sizeof(T*));
    if(e != cudaSuccess) {
        printf("Error allocating dA_array: %s\n", cudaGetErrorString(e));
    }
	cudaMemcpy(dA_array, hA_array.data(), batch_size * sizeof(T*), cudaMemcpyHostToDevice);


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
}

/******************************************************************************/
/*                 CODES FOR NON-UNIFORM BATCH GENERATION                     */
/******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
template<typename T>
__global__
void init_matrix_kernel_vbatched(int *m, int *n, T** dA_array, int *ldda)
{
	const int batchid = blockIdx.z;
	const int gtx     = blockIdx.x * blockDim.x + threadIdx.x;
	const int gty     = blockIdx.y * blockDim.y + threadIdx.y;

	// local problem dimensions
	const int m_     = m[batchid];
	const int n_     = n[batchid];
	const int ldda_  = ldda[batchid];

    init_matrix_device<T>(m_, n_, dA_array[batchid], ldda_, gtx, gty);
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void init_matrix_vbatched(int max_m, int max_n, int *m, int *n, T** dA_array, int *ldda, int batch_size)
{
	dim3 threads(THREAD_X, THREAD_Y, 1);

	const int gridx = (max_m + (THREAD_X-1)) / THREAD_X;
	const int gridy = (max_n + (THREAD_Y-1)) / THREAD_Y;
	// grid-z dimension has a max value of ~65k
	// for batches larger than this, we should use a loop
	const int max_batch = 60000;
	for(int i = 0; i < batch_size; i+=max_batch) {
		int gridz = min(max_batch, batch_size-i);
		dim3 grid(gridx, gridy, gridz);
		init_matrix_kernel_vbatched<<<grid, threads, 0, 0>>>(m, n, dA_array+i, ldda);
	}
}

////////////////////////////////////////////////////////////////////////////////
template<typename T>
void init_matrix_vbatched_test(int max_m, int max_n, int batch_size)
{
    printf("Uniform Batch Generation Test\n");
    printf("=============================\n");
    printf("Initializing a non-uniform batch of %d matrices -- max. rows = %d, max_cols = %d\n", batch_size, max_m, max_n);

    // init dimensions randomly, and compute the total memory required
    std::vector<int> m(batch_size);
    std::vector<int> n(batch_size);
    int* ldda  = m.data();
    size_t  sizeA = 0;
    for(int i = 0; i < batch_size; i++) {
        m[i] = 1 + ( rand() % max_m );
        n[i] = 1 + ( rand() % max_n );

        sizeA += ldda[i] * n[i] * sizeof(T);
    }

	// alloc matrices on GPU
	printf("Data size = %.2f GB\n", sizeA / (1024. * 1024. * 1024.));
    T* dA = NULL;
	cudaError_t e = cudaMalloc((void**)&dA, sizeA);
	if(e != cudaSuccess) {
	    printf("Error allocating dA: %s\n", cudaGetErrorString(e));
	}

	// setup pointer array on cpu
	std::vector<T*>hA_array(batch_size);
	hA_array[0] = dA;
	for(int i = 1; i < hA_array.size(); i++) {
	    hA_array[i] = hA_array[i-1] + ldda[i-1] * n[i-1];
	}

    // ptr array on the GPU
    T** dA_array = NULL;
	e = cudaMalloc((void**)&dA_array, batch_size * sizeof(T*));
    if(e != cudaSuccess) {
        printf("Error allocating dA_array: %s\n", cudaGetErrorString(e));
    }
	cudaMemcpy(dA_array, hA_array.data(), batch_size * sizeof(double*), cudaMemcpyHostToDevice);

    // alloc dimensions arrays on GPU
    int *d_m = NULL, *d_n=NULL;
    e = cudaMalloc((void**)&d_m, batch_size * sizeof(int)); if(e != cudaSuccess) {printf("Error allocating d_m: %s\n", cudaGetErrorString(e));}
    e = cudaMalloc((void**)&d_n, batch_size * sizeof(int)); if(e != cudaSuccess) {printf("Error allocating d_n: %s\n", cudaGetErrorString(e));}

    cudaMemcpy(d_m, m.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // leading dimensions on GPU
    int* d_ldda = d_m;

    // for timing
    cudaEvent_t start, stop;
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop,  0);

    // warmup run
	init_matrix_vbatched(max_m, max_n, d_m, d_n, dA_array, d_ldda, batch_size);

	// launch kernel
	cudaEventRecord(start);
	for(int i = 0; i < NITER; i++) {
		init_matrix_vbatched(max_m, max_n, d_m, d_n, dA_array, d_ldda, batch_size);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// print time
	float ms = 0.;
	cudaEventElapsedTime(&ms,start,stop);
	ms /= (float)NITER;
	float bw = (sizeA / 1e9) / (ms / 1000.);
	printf("Initialization time: %.2f ms -- %.2f GB/s\n", ms, bw);

    if(d_m      != NULL) cudaFree(d_m);
    if(d_n      != NULL) cudaFree(d_n);
    if(dA       != NULL) cudaFree(dA);
    if(dA_array != NULL) cudaFree(dA_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

	// fixed size generation
	init_matrix_batched_test<double>(m, n, batch_size);
	printf("\n\n");

	// variable size generation
	init_matrix_vbatched_test<double>(m, n, batch_size);
	printf("\n");

	return 0;
}

