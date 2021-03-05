CC = dpcpp
FLAGS = -std=c++17  -O2
OBJ = run_matmul
SOURCES = matmul.cpp
#DPCPP_OPTS=-I/glob/development-tools/versions/oneapi/gold/inteloneapi/mkl/latest/include -mkl #-fsycl-device-code-split=per_kernel

all:
	$(CC) $(FLAGS) -o $(OBJ) $(SOURCES) $(DPCPP_OPTS)
clean:
	rm -rf $(OBJ)

