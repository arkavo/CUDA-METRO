#include <stdio.h>

void findFirstNeighbors(int* matrix, int N, int index,FILE *fptr) {
    // Calculate row and column of the given index
    int row = index / N;
    int col = index % N;
int i;
    // Define the offsets for N4 2-2-4-2 neighbors
    int offsets_rows[] = {-1,1};
  	int offsets_coloumn[] = {-1,1};
   // printf("First neighbors (N4) of element at index %d:\n", index);

    // Iterate over the offsets and find the neighboring indices
    for ( i = 0; i < 4; i++) {
    	int neighborRow = row  ;
        int neighborCol = col ;
    	
    	if(row<N/2)
    	{
		
     
     
     
        
                if(i==0)
        {
          	neighborRow=neighborRow +(N/2);
         	neighborCol=neighborCol-1;

		}
        else if(i==1)
        {
          	neighborRow=neighborRow +(N/2);
         	neighborCol=neighborCol;
		}
		 else if(i==2)
        {
        	if(neighborRow+1>=(N/2))
        	{
        		neighborRow=N/2;
			}
			else{
				
	          	neighborRow=neighborRow +(N/2)+1;

			}
        
         	neighborCol=neighborCol-1;

		}
		 else if(i==3)
        {
		if(neighborRow+1>=(N/2))
        	{
        		neighborRow=N/2;
			}
			else{
	
	          	neighborRow=neighborRow +(N/2)+1;
			}
                 	neighborCol=neighborCol;
		}
 
        
    
          if(neighborCol<0)
        {
        	neighborCol=neighborCol+N;
		}
		 if(neighborCol>=N)
        {
        	neighborCol=neighborCol-N;
		}
	}
	
	
	

	
	if(row==N/2)
	{
		
	     if(i==0)
        {
          	neighborRow= (N/2)-1;
         	neighborCol=neighborCol;

		}
        else if(i==1)
        {
          	neighborRow=(N/2);
         	neighborCol=neighborCol+1;
		}
		 else if(i==2)
        {
          	neighborRow=0;
         	neighborCol=neighborCol;

		}
		 else if(i==3)
        {
          	neighborRow=0;
         	neighborCol=neighborCol+1;
		}
 
        
    
          if(neighborCol<0)
        {
        	neighborCol=neighborCol+N;
		}
		 if(neighborCol>=N)
        {
        	neighborCol=neighborCol-N;
		}	
		
	}
      
	  	
	if(row>N/2)
	{
		
	     if(i==0)
        {
          	neighborRow=neighborRow - (N/2)-1;
         	neighborCol=neighborCol;

		}
        else if(i==1)
        {
          	neighborRow=neighborRow-(N/2)-1;
         	neighborCol=neighborCol+1;
		}
		 else if(i==2)
        {
          	neighborRow=neighborRow-(N/2);
         	neighborCol=neighborCol;

		}
		 else if(i==3)
        {
          	neighborRow=neighborRow-(N/2);
         	neighborCol=neighborCol+1;
		}
 
        
    
          if(neighborCol<0)
        {
        	neighborCol=neighborCol+N;
		}
		 if(neighborCol>=N)
        {
        	neighborCol=neighborCol-N;
		}	
		
	}
	  
	  
	    int neighborIndex = neighborRow * N + neighborCol ;
        fprintf(fptr,"%d ", neighborIndex);
        if(i<1)
        {
        //	fprintf(fptr,",");
		}
    
}
    fprintf(fptr,"\n");
}

int main() {
    int N = 128; // Size of the matrix
    int matrix[N * N];
int i;
    FILE *fptr;
   fptr= fopen("C:\\Users\\deric\\OneDrive - Indian Institute of Science\\Desktop\\Major_Project\\Codes_Neigh_Generic\\N2-mapping\\output5.txt","w"); 
         if(fptr == NULL)
   {
      printf("Error!");   
      exit(1);             
   }
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

