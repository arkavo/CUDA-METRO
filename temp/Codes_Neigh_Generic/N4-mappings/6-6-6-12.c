#include <stdio.h>



void findFirstNeighbors(int* matrix, int N, int index,FILE *fptr) {
    // Calculate row and column of the given index
    
   

    int row = index / N;
    int col = index % N;
int i;
    // Define the offsets for N 4-4-4-8 neighbors
    int offsets_row[] = {-1,1};
    int offsets_coloumn[] = {-1,1};

  //  printf("First neighbors (N4) of element at index %d:\n", index);

    // Iterate over the offsets and find the neighboring indices
    for ( i = 0; i < 12; i++) {
        
        int neighborRow = row ;
        int neighborCol = col ;
        if(i==0)
        {
          	neighborRow=neighborRow-2;
         	neighborCol=neighborCol-3;

		}
        else if(i==1)
        {
          	neighborRow=neighborRow-3;
         	neighborCol=neighborCol-1;
		}
		 else if(i==2)
        {
          	neighborRow=neighborRow-3;
         	neighborCol=neighborCol-2;
		}
		 else if(i==3)
        {
          	neighborRow=neighborRow+1;
         	neighborCol=neighborCol+3;		}
        	 else if(i==4)
        {
        	neighborCol=neighborCol +2;
        	neighborRow=neighborRow -1;
		}
        	 else if(i==5)
        {
          	neighborRow=neighborRow-2;
         	neighborCol=neighborCol+1;		}
        if(i==6)
        {
          	neighborRow=neighborRow+2;
         	neighborCol=neighborCol-1;

		}
        else if(i==7)
        {
          	neighborRow=neighborRow+1;
         	neighborCol=neighborCol-2;
		}
		 else if(i==8)
        {
          	neighborRow=neighborRow-1;
         	neighborCol=neighborCol-3;
		}
		 else if(i==9)
        {
          	neighborRow=neighborRow+3;
         	neighborCol=neighborCol+2;		}
        	 else if(i==10)
        {
        	neighborCol=neighborCol + 1;
        	neighborRow=neighborRow +3;
		}
        	 else if(i==11)
        {
          	neighborRow=neighborRow+2;
         	neighborCol=neighborCol+3;		}
        
        
          if(neighborCol<0)
        {
        	neighborCol=neighborCol+N;
		}
		 if(neighborCol>=N)
        {
        	neighborCol=neighborCol-N;
		}
				
	    if(neighborRow<0)
        {
        	neighborRow=neighborRow+N;
		}
		 if(neighborRow>=N)
        {
        	neighborRow=neighborRow-N;
		}
		
        int neighborIndex = (neighborRow * N + neighborCol );
        fprintf(fptr,"%d ", neighborIndex);
        if(i<11){
        			fprintf(fptr,",");

		}
    }

    fprintf(fptr,"\n");
}

int main() {
    int N = 512; // Size of the matrix
    int matrix[N * N];
      FILE *fptr;
   fptr= fopen("C:\\Users\\deric\\OneDrive - Indian Institute of Science\\Desktop\\Major_Project\\Codes_Neigh_Generic\\N4-mappings\\output6.csv","w"); 
         if(fptr == NULL)
   {
      printf("Error!");   
      exit(1);             
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

