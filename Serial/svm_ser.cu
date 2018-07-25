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
const char *filenameTrainClassSizes = "class_sizes_train_1024.csv";
const char *filenameTestClassSizes = "class_sizes_train_1024.csv";
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

	result[0] = 0.0;
	
	for(int k = 0; k<c; k++){
		result[0] += xi[k]*W[k];	
	}
	
}

void updateWeights(float *W, float n, float y, float alpha, float *x, int violatesMC){
	for(int i = 0; i<c; i++){
		W[i] = (1.0 - n*alpha)*W[i] + n*y*x[i]*violatesMC;
	}
}

void pegaFit(float *X, float *y, float *W, unsigned int T, float alpha, unsigned int r,float *runTime,StopWatchInterface *timer){

	float *xi = (float *)malloc(c * sizeof(float)); 
	float yi;
	float *result = (float *)malloc(sizeof(float));
	float ni;
	srand(time(NULL));
	float time= sdkGetTimerValue(&timer);
	for(int i = 1; i <= T; i++){
	
		ni = 1.0/(alpha*(i));
		
		int idx = rand() % r;
		//bringing in current x and y
		for(int k = 0; k<c; k++){
			xi[k] = X[idx*c + k];
		}
		yi = y[idx];
		
		//if violates margin constraint 0.0 and 1.0 are multiplier to nulify second factor
		
		dotProd(xi, W, result);
		
		if(!(yi*(result[0]) >= 1)){
			updateWeights(W, ni, yi, alpha, xi, 1.0);
		}
		else{
			updateWeights(W, ni, yi, alpha, xi, 0.0);
		}
	}
	runTime[0] = sdkGetTimerValue(&timer) - time;
	//printf("weight[0]: %f\n",W[0]);
	
}

void voting(float *xi, float *W, float *voteMat){


	float *dot_calc = (float *)malloc(sizeof(float));
	
	for(int j = 1; j<numClasses; j++){
		for(int k = 0; k<j; k++){
		
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
	
	/*for(int loop1 =0; loop1<numClasses;loop1++){
		printf("%d\t",vote_count[loop1]);
	}
	printf("\n");
	*/
	return max_ind;
}

int main(int argc, char **argv){
	
	//-----------------------DATA READ TRAIN------------------------
	
	FILE* stream = fopen(filenameTrainClassSizes, "r");
	
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
	
	//-----------------------DATA READ TEST------------------------
	stream = fopen(filenameTestClassSizes, "r");
	
	//class sizes holds the number of rows in each class, 
	//as well as what row number the class begins
	int *classSizesTe = (int *)malloc(numClasses * 2 * sizeof(int)); 
	
    
	i = 0;
	int numRecordsTe=0;
    while (fgets(line, line_len, stream))
	{
	
		classSizesTe[i*2 + 0] = numRecordsTe;
		classSizesTe[i*2 + 1] = atoi(line);
		numRecordsTe += classSizesTe[i*2+1];
		i++;
    }
	
	stream = fopen(filenameTest, "r");
	float *dataTe = (float *)malloc(numRecordsTe * c * sizeof(float));
		
	i = 0;
	char** tokensTe;
	while (fgets(line, line_len, stream))
	{
	    tokensTe = str_split(line, ',');
	    
	    for(int l = 0; l<c; l++){
			dataTe[i*c + l] = atof(tokensTe[l]);
		}
		i++;
	}
	//-----------------------WEIGHT MATRIX---------------------
	
	
	float *W = (float *)malloc(numClasses * numClasses * c * sizeof(float));
	float *voteMat = (float *)malloc(numClasses * numClasses * sizeof(float));
	float *time = (float *)malloc(sizeof(float));
	time[0] = 0;
	//-----------------------SERIAL PEGASOS--------------------- 
	
	StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
	
	for(int j = 1; j<numClasses; j++){
		for(int k = 0; k<j; k++){
			
		
			int trainRows = classSizes[j*2 + 1] + classSizes[k*2 + 1];
			float *train_X = (float *)malloc(trainRows * c * sizeof(float));
			float *train_y = (float *)malloc(trainRows * sizeof(float));
			
			getModelData(data, train_X, train_y, classSizes, j, k);
			
			pegaFit(train_X, train_y, &W[j*numClasses*c+k*c+0], T, 0.0001/*alpha*/, trainRows, time, timer);
		}
	}
	
	sdkStopTimer(&timer);
	float tot_time = sdkGetTimerValue(&timer);
    printf("Total time      :\t%f(ms)\n", tot_time);
    printf("Processing time :\t%f(ms)\n", time[0]);
    printf("Overheads time  :\t%f(ms)\n", tot_time-time[0]);
    sdkDeleteTimer(&timer);
	
	int dec=0;
	int numCorr=0;
	for(int j=0; j<numRecordsTe; j++){
		voting(&dataTe[j*c], W, voteMat);
		//printf("rec:%d\t",j);
		dec = decision(voteMat);
		
		for(int k=0;k<numClasses;k++){
			if(j>classSizesTe[k*2+0] && j < (classSizesTe[k*2 + 0]+classSizesTe[k*2 + 1]) ){
			
				if(k == dec){
					numCorr++;
				} else {
					//printf("Dec: %d \tExpected: %d\n",dec,k);
					/*for(int loop1 =0; loop1<numClasses;loop1++){
						for(int loop2=0;loop2<numClasses;loop2++)
							printf("%f\t",voteMat[loop1*numClasses+loop2]);
						printf("\n");
					}*/
							
				}
				
				break;
			}
		}
	}
	float avg = (numCorr/(numRecordsTe+0.0))*100;
	printf("Correct Percentage: %f\n",avg);
}
