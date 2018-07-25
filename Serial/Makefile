INC="./inc/"
NVCCFLAGS=-I$(INC)
NVCC=nvcc

all: svm

svm: svm_ser.cu
	$(NVCC) $(NVCCFLAGS) svm_ser.cu -o a

clean:
	rm svm
