SM?=80
FLAGS=-gencode arch=compute_$(SM),code=sm_$(SM)

all:
	nvcc kaust_init_matrix.cu $(FLAGS) -o kaust_init_matrix

clean:
	rm -f kaust_init_matrix
