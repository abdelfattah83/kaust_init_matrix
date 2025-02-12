This is a simple framework for initializing uniform and non-uniform batches of matrices. 

For uniform matrices, the user specifies the matrix dimensions. For non-uniform batches, the user specifies the maximum rows/columns across the batch. 

Simply build using `make`, only `nvcc` is required.

You can run:  `./kaust_init_matrix <batch-size> <M> <N>`

By default, batch-size = 1000, M = 50, N = 50

You can also play with the kernel thread configuration: `THREAD_X` and `THREAD_Y`
