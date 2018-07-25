//#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


//------------------DATA SETUP-------------------

const unsigned int numClasses = 10;
const unsigned int T = 100000;
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
            //assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        //assert(idx == count - 1);
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

void pegaFit(float *X, float *y, float *W, unsigned int T, float alpha, unsigned int r, int myRank){

    float *xi = (float *)malloc(c * sizeof(float));
    float yi;
    float *result = (float *)malloc(sizeof(float));
    float ni;
    srand(time(NULL));
    //float time= sdkGetTimerValue(&timer);
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
        /*if(myRank==0 && i%10 == 0){
        printf("xi[0]: %.3f\tw[0]:%.3f\n",xi[0],W[0]);
    }*/

    }

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

    int vote_count[numClasses];
    for(int i=0;i<numClasses;i++)
        vote_count[i]=0;

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

int main(int argc, char *argv[]){
    int nproces, myrank, next, prev, tag=1;
    //char token[MSG_SZ];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproces);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //printf("From process %d out of %d, Hello World!\n",myrank, nproces);

    /*-----------------------------------------------------------------------------------------------------------------------------
    MegaDos- Room of angry women technique (no communication)
    -----------------------------------------------------------------------------------------------------------------------------*/

    /*--------------------------------------------SECTION 1------------------------------------------------------------------------
    All nodes generate the rank indeces - this is the values that the node will be training
    for eg:
    rankIndeces[myRank][0] = 1
    rankIndeces[myRank][1] = 4
    node with myRank will be training the weights comparing 1 and 4
    (working)
    -----------------------------------------------------------------------------------------------------------------------------*/
    double totalTime = MPI_Wtime();
    double overhead = 0;
    double procesingTime = 0;
    double tempTime =0;
    int rankIndeces[45][2]={0};

    int counterPos = 0;
    for(int i = 1; i<numClasses; i++){
        for(int j = 0; j<i; j++){
            rankIndeces[counterPos+j][0] = i;
            rankIndeces[counterPos+j][1] = j;
        }
        counterPos+=i;
    }

    //class sizes holds the number of rows in each class,
    //as well as what row number the class begins
    int *classSizes = (int *)malloc(numClasses * 2 * sizeof(int));
    int numRecords=0;
    FILE* stream = fopen(filenameTrainClassSizes, "r");
    char line[line_len];
    unsigned int readCounter = 0;

    //--------------------------------------------SECTION 1------------------------------------------------------------------------

    /*--------------------------------------------SECTION 2------------------------------------------------------------------------
    rank 0 reads in0.822516 all the data and broadcasts it to the rest of the nodes (working)
    -----------------------------------------------------------------------------------------------------------------------------*/

    if(myrank ==0){
        while (fgets(line, line_len, stream))
        {

            classSizes[readCounter*2 + 0] = numRecords;
            classSizes[readCounter*2 + 1] = atoi(line);
            numRecords += classSizes[readCounter*2+1];
            readCounter++;
        }
    }


    tempTime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(classSizes, numClasses * 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numRecords, 1, MPI_INT, 0, MPI_COMM_WORLD);
    overhead += MPI_Wtime()-tempTime;


    float *data = (float *)malloc(numRecords * c * sizeof(float));

    if(myrank == 0){
        stream = fopen(filenameTrain, "r");

        readCounter = 0;
        char** tokens;
        while (fgets(line, line_len, stream))
        {
            tokens = str_split(line, ',');

            for(int l = 0; l<c; l++){
                data[readCounter*c + l] = atof(tokens[l]);

            }
            readCounter++;
        }
    }
    tempTime = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(data, numRecords * c, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    overhead += MPI_Wtime()-tempTime;
    //--------------------------------------------SECTION 2------------------------------------------------------------------------


    /*--------------------------------------------SECTION 3------------------------------------------------------------------------
    All nodes create the full weight array(W) and a weight reduction array (reducVar)
    All nodes will train their portion of W, (accessed with rankIndeces)
    -----------------------------------------------------------------------------------------------------------------------------*/

    float *W = (float *)malloc(numClasses * numClasses * c * sizeof(float));
    float *reducVar = (float *)malloc(numClasses * numClasses * c * sizeof(float));

    int j=rankIndeces[myrank][0];
    int k=rankIndeces[myrank][1];

    int trainRows = classSizes[j*2 + 1] + classSizes[k*2 + 1];
    float *train_X = (float *)malloc(trainRows * c * sizeof(float));
    float *train_y = (float *)malloc(trainRows * sizeof(float));

    getModelData(data, train_X, train_y, classSizes, j, k);

    tempTime = MPI_Wtime();
    pegaFit(train_X, train_y, &W[j*numClasses*c + k*c+0], T, 0.0001, trainRows,myrank);
    procesingTime = MPI_Wtime() - tempTime;
    //W[j*numClasses*c+ k*c]

    //--------------------------------------------SECTION 3------------------------------------------------------------------------
    /*
    if (myrank == 0){
    printf("Thread:%d\n",myrank);
    for(int i=0;i<numClasses;i++){
    for(int j=0;j<numClasses;j++){
    printf("%f\t",W[i*numClasses*c + j*c]);
}
printf("\n");
}
printf("\n");
}*/




/*--------------------------------------------SECTION 4------------------------------------------------------------------------
only myrank == 0 will get it's reducVar populated using the reduce function.
-----------------------------------------------------------------------------------------------------------------------------*/
tempTime = MPI_Wtime();
MPI_Reduce(W, reducVar, numClasses * numClasses * c, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
overhead += MPI_Wtime()-tempTime;
/*
if(myrank==0)
for(int i=0;i<numClasses;i++){
for(int j=0;j<numClasses;j++){
printf("%f\t",reducVar[i*numClasses*c + j*c]);
}
printf("\n");
}*/

//--------------------------------------------SECTION 4------------------------------------------------------------------------

/*--------------------------------------------SECTION 5------------------------------------------------------------------------
Only myrank == 0 will perform the tests
-----------------------------------------------------------------------------------------------------------------------------*/

totalTime = MPI_Wtime()-totalTime;
if (myrank == 0){
    float *voteMat = (float *)malloc(numClasses * numClasses * sizeof(float));
    int dec=0;
    int numCorr=0;

    for(int j=0; j<numRecords; j++){
        voting(&data[j*c], reducVar, voteMat);


        dec = decision(voteMat);

        for(int k=0;k<numClasses;k++){
            if(j>classSizes[k*2+0] && j < (classSizes[k*2 + 0]+classSizes[k*2 + 1]) ){

                if(k == dec)
                    numCorr++;
                break;
            }
        }
    }

    /*for(int m=0;m<numClasses;m++){
        for(int n=0;n<numClasses;n++)
            printf("%f\t", reducVar[m*numClasses*c+n*c]);
        printf("\n");
    }*/

    printf("Processing Time:%f\n",procesingTime);
    printf("Overhead Time  :%f\n",overhead);
    printf("Total Time     :%f\n",totalTime);
    printf("Correct Percentage: %f\n", (numCorr/(numRecords+0.0))*100 );
}
//--------------------------------------------SECTION 5------------------------------------------------------------------------


MPI_Finalize();
return 0;
}
