//#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"

// CUDA helper functions
#include "helper_cuda.h" 

//------------------DATA SETUP-------------------

const unsigned int numClasses = 10;
const unsigned int T = 10000;
const char *filenameTrain = "train_data_1024.csv";
const char *filenameTest = "train_data_1024.csv";
const char *classSizeFile ="class_sizes_train_1024.csv";
const unsigned int c = 1024;
const unsigned int line_len = 35000;



//-----------------------------------------------

//https://stackoverflow.com/questions/9210528/split-string-with-delimiters-in-c
char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = (char**)malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

/*
X, y enter the method as null pointers.
-The necessary data is extracted and stored in X.
-The y values are also generated according to the classSizes.
*/
void getModelData(float *data, float *X, float *y, int *classSizes, int cl1_id, int cl2_id){
	
	int cl1_start_idx = classSizes[cl1_id*2 + 0];
	int cl1_end_idx = cl1_start_idx + classSizes[cl1_id*2 + 1];
	int cl2_start_idx = classSizes[cl2_id*2 + 0];
	int cl2_end_idx = cl2_start_idx + classSizes[cl2_id*2 + 1];
	int k = 0;
	
	for(int i = cl1_start_idx; i<cl1_end_idx; i++){
		for(int j = 0; j<c; j++){
			X[k*c + j] = data[i*c + j];
		}
		y[k] = 1;
		k++;
	}
	
	for(int i = cl2_start_idx; i<cl2_end_idx; i++){
		for(int j = 0; j<c; j++){
			X[k*c + j] = data[i*c + j];
		}
		y[k] = -1;
		k++;
	}
}

void dotProd(float *xi, float *W, float *result){

	//result[0] = 0.0;
	
	for(int k = 0; k<c; k++){
		result[0] += xi[k]*W[k];	
	}
	
}

__global__ void dotprodPar(float *X, float *W, float *ni , float *y, int *pos){
	
	__shared__ float partialSum[c];
	__shared__ int mult;
	
	unsigned int idx = threadIdx.y+threadIdx.x;
	
	//printf("ni[0]:%f\n",ni[0]);
	
	partialSum[idx] = X[(pos[0])*c+idx] * W[idx];
	
	__syncthreads();
	
	for(int stride = blockDim.x/2; stride >= 1; stride = stride >> 1){
		__syncthreads();	
		if(idx < stride){
			partialSum[idx] += partialSum[idx + stride];
		}
	}
	
	__syncthreads();
	
	if(idx == 0){
		if(!(y[pos[0]]*(partialSum[0]) >= 1))
			mult=1;
		else
			mult=0;
	}
	__syncthreads();
	
	W[idx] = (1.0 - ni[0]*0.0001)*W[idx] + ni[0]*y[pos[0]]*X[(pos[0])*c+idx]*mult;
	
	__syncthreads();
	
	if(idx == 0){
		ni[0] = 1.0/(0.0001*pos[1]);
	}
	
		
}


void pegaFit(float *X, float *y, float *W, unsigned int T, float alpha, int r, float *runTime,StopWatchInterface *timer){

	float *result = (float *)malloc(c * sizeof(float));
	float *ni = (float *)malloc(sizeof(float));
	srand(time(NULL));
	
	//parrallel setup
	
    float *x_par = NULL;
    checkCudaErrors(cudaMalloc((void **) &x_par, c * r * sizeof(float)));
	float *w_par = NULL;
	checkCudaErrors(cudaMalloc((void **) &w_par, c * sizeof(float)));
	float *ni_par = NULL;
    checkCudaErrors(cudaMalloc((void **) &ni_par, sizeof(float)));
	float *yi_par = NULL;
	checkCudaErrors(cudaMalloc((void **) &yi_par, r * sizeof(float)));
	//int *r_par = NULL;
	//checkCudaErrors(cudaMalloc((void **) &r_par, sizeof(int)));
	//end setup
	
	ni[0] = 1.0/(alpha*1);
	
	int *pos = (int *)malloc(2*sizeof(int));
	int *pos_par = NULL;
	cudaMalloc((void **) &pos_par, 2*sizeof(int));
	/*
	malloc the full X and Y data
	generate random guess inside dotprodpar
	paralelise voting & decision
	make accuracy a method
	*/
	cudaMemcpy(x_par, X, c * r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yi_par, y, r * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(w_par, W, c * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ni_par, ni, sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(r_par, &r, sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy()
	/*
	StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    */
    float time= sdkGetTimerValue(&timer);
	for(int i = 1; i <= T; i++){
		
		pos[0] = rand()%r;
		pos[1] = i;
		//printf("%d\n",pos[0]);
		cudaMemcpy(pos_par, pos, 2*sizeof(int), cudaMemcpyHostToDevice);
		dotprodPar<<<1, c>>>(x_par, w_par, ni_par,yi_par,pos_par);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	runTime[0] = sdkGetTimerValue(&timer) - time;
	/*
	sdkStopTimer(&timer);
    runTime[0] += sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
	*/
	cudaMemcpy(W, w_par, c * sizeof(float), cudaMemcpyDeviceToHost);
	
	//printf("weight[0]: %f\n",W[0]);
	
}

void voting(float *xi, float *W, float *voteMat){


	float *dot_calc = (float *)malloc(sizeof(float));
	
	for(int j = 1; j<numClasses; j++){
		for(int k = 0; k<j; k++){
			
			dot_calc[0] = 0;
			dotProd(xi, &W[j*numClasses*c + k*c+0], dot_calc);
			if (dot_calc[0] < 0){
				voteMat[j*numClasses+k+0] = -1;
			} else {
				voteMat[j*numClasses+k+0] = 1;
			}
		}
		
	}
	
	
}

int decision(float *voteMat){

	int vote_count[numClasses] = {0};
	
	for(int j = 1; j<numClasses; j++){
		for(int k = 0; k<j; k++){
			if (voteMat[j*numClasses+k] < 0)
				vote_count[k] += 1;
			else
				vote_count[j] += 1;
		}
		
	}
	int max_ind=0;
	for(int i=0; i < numClasses;i++){
		if(vote_count[i] > vote_count[max_ind])
			max_ind = i;
	}
	return max_ind;
}

int numCorrect(int numRecords, float *data, float *W, int *classSizes){
	float *voteMat = (float *)malloc(numClasses * numClasses * sizeof(float));
	int numCorr= 0;
	int dec=0;
	
	for(int j=0; j<numRecords; j++){
		voting(&data[j*c], W, voteMat);
		dec = decision(voteMat);
		
		for(int k=0;k<numClasses;k++){
			if(j>classSizes[k*2+0] && j < (classSizes[k*2 + 0]+classSizes[k*2 + 1]) ){
			
				if(k == dec){
					numCorr++;
				}
				break;
			}
		}
	}
	return numCorr;
}

int main(int argc, char **argv){
	
	//-----------------------DATA READ------------------------
	
	FILE* stream = fopen(classSizeFile, "r");
	
	//class sizes holds the number of rows in each class, 
	//as well as what row number the class begins
	int *classSizes = (int *)malloc(numClasses * 2 * sizeof(int)); 
	
    char line[line_len];
	unsigned int i = 0;
	int numRecords=0;
    while (fgets(line, line_len, stream))
	{
	
		classSizes[i*2 + 0] = numRecords;
		classSizes[i*2 + 1] = atoi(line);
		numRecords += classSizes[i*2+1];
		i++;
    }
	
	stream = fopen(filenameTrain, "r");
	float *data = (float *)malloc(numRecords * c * sizeof(float));
		
	i = 0;
	char** tokens;
	while (fgets(line, line_len, stream))
	{
	    tokens = str_split(line, ',');
	    
	    for(int l = 0; l<c; l++){
			data[i*c + l] = atof(tokens[l]);
			
		}
		i++;
	}
	
	
	//-----------------------WEIGHT MATRIX---------------------
	
	
	float *W = (float *)malloc(numClasses * numClasses * c * sizeof(float));
	float *time = (float *)malloc(sizeof(float));
	time[0] = 0;
	//-----------------------SERIAL PEGASOS--------------------- 
	/*
	The following for loop extracts the relevant X and Y data.
	This data is passed into pegafit along with 
	the single weight vector for that particular
	comparison.
	Pegafit trains the weights for the particular 
	comparison.
	*/
	StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    
    
	float alpha = 0.0001;
	
	for(int j = 1; j<numClasses; j++){
		for(int k = 0; k<j; k++){
			
		
			int trainRows = classSizes[j*2 + 1] + classSizes[k*2 + 1];
			float *train_X = (float *)malloc(trainRows * c * sizeof(float));
			float *train_y = (float *)malloc(trainRows * sizeof(float));
			
			getModelData(data, train_X, train_y, classSizes, j, k);
			
			pegaFit(train_X, train_y, &W[j*numClasses*c+k*c+0], T, alpha, trainRows, time,timer);
			
		}
	}
	
	sdkStopTimer(&timer);
	float tot_time = sdkGetTimerValue(&timer);
    printf("Total time      :\t%f(ms)\n", tot_time);
    printf("Processing time :\t%f(ms)\n", time[0]);
    printf("Overheads time  :\t%f(ms)\n", tot_time-time[0]);
    sdkDeleteTimer(&timer);
	
	stream = fopen(filenameTrain, "r");
	i = 0;
	while (fgets(line, line_len, stream))
	{
	    tokens = str_split(line, ',');
	    
	    for(int l = 0; l<c; l++){
			data[i*c + l] = atof(tokens[l]);
			
		}
		i++;
	}
	
	
	int numCorr=numCorrect(numRecords, data, W,classSizes);
	
	float avg = (numCorr/(numRecords+0.0))*100;
	printf("Correct Percentage: %f\n",avg);
}
