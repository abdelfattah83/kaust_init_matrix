all:
	nvcc kaust_init_matrix.cu -o kaust_init_matrix

clean:
	rm -f kaust_init_matrix
