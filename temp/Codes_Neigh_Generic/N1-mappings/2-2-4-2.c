#include <stdio.h>

void findFirstNeighbors(int* matrix, int N, int index,FILE *fptr) {
    // Calculate row and column of the given index
    int row = index / N;
    int col = index % N;
int i;
    // Define the offsets for N4 2-2-4-2 neighbors
    int offsets[] = {-1,1};

    //printf("First neighbors (N4) of element at index %d:\n", index);

    // Iterate over the offsets and find the neighboring indices
    for ( i = 0; i < 2; i++) {
        int neighborRow = (row + offsets[i] );
        if(neighborRow<0)
        {
        	neighborRow=neighborRow+N;
		}
		 if(neighborRow>=N)
        {
        	neighborRow=neighborRow-N;
		}
        int neighborCol = col ;
        int neighborIndex = neighborRow * N + neighborCol ;
        fprintf(fptr,"%d ", neighborIndex);
        if(i<1)
        {
        	fprintf(fptr,",");
		}
    }

    fprintf(fptr,"\n");
}

int main() {
    int N = 512; // Size of the matrix
    int matrix[N * N];


   FILE *fptr;
   
   fptr= fopen("C:\\Users\\deric\\OneDrive - Indian Institute of Science\\Desktop\\Major_Project\\Codes_Neigh_Generic\\output2.txt","w"); 	
	if(fptr == NULL)
   {
      printf("Error!");   
      //exit(1);             
   }
int i;
    // Initialize the matrix with some values
    for (i = 0; i < N * N; i++) {
        matrix[i] = i + 1;
    }

    // Find first neighbors (N4) for each element
    for ( i = 0; i < N*N; i++) {
        findFirstNeighbors(matrix, N, i,fptr);
    }

    return 0;
}

