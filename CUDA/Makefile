INC="./inc/"
NVCCFLAGS=-I$(INC)
NVCC=nvcc

all: svm

svm: SVM.cu
	$(NVCC) $(NVCCFLAGS) SVM.cu -o a

clean:
	rm svm
