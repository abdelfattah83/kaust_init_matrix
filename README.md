This is a simple framework for initializing a batch of matrices having the same `M x N` dimensions.

Simply build using `make`, only `nvcc` is required.

By default, the code initializes 1000 matrices of size 50 x 50

You can also run:  `./kaust_init_matrix <batch-size> <M> <N>`

You can also play with the kernel thread configuration: `THREAD_X` and `THREAD_Y`
