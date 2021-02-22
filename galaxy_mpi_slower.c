// module load OpenMPI
// export OMPI_MCA_btl_openib_allow_ib=1 
// For MPI programs, compile with
//    mpicc -O3 -o galaxy galaxy.c -lm
//
// and run with e.g. 100 cores
//    srun -n 100 --mem=1G -t 2:00:00 ./galaxy RealGalaxies_100k_arcmin.txt SyntheticGalaxies_100k_arcmin.txt omega.txt   

#include <mpi.h>
//#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int MemoryAllocatedCPU = 0L;
int parseargs_readinput(int argc, char *argv[]);
void calc_angles_RR(int rows, int start, float *rand_d, float *rand_r, long int *hist);
void calc_angles_DR(int rows, int start, float *real_d, float *real_r, float *rand_d, float *rand_r, long int *hist);
void calc_angles_DD(int rows, int start, float *real_d, float *real_r, long int *hist);

int main(int argc, char* argv[]) 
{
    struct timeval _ttime;
    struct timezone _tzone;
	int world_size;
	int rank;
	
    pif = acosf(-1.0f);
	MPI_Status status[4];
	MPI_Init(NULL,NULL);

    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
	
	//Get the number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	
	//Get the rank of the processes
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
    // store right ascension and declination for real galaxies here
    // Note: indices run from 0 to 99999 = 100000-1: realrasc[0] -> realrasc[99999] 
    // realrasc[100000] is out of bounds for allocated memory!
    real_rasc        = (float *)calloc(100000L, sizeof(float));
    real_decl        = (float *)calloc(100000L, sizeof(float));

    // store right ascension and declination for synthetic random galaxies here
    rand_rasc        = (float *)calloc(100000L, sizeof(float));
    rand_decl        = (float *)calloc(100000L, sizeof(float));

    MemoryAllocatedCPU += 10L*100000L*sizeof(float);
	
	if ( rank == 0)
	{
		if ( parseargs_readinput(argc, argv) != 0 ) 
		{
			printf("   Program stopped.\n");return(0);
		}
		printf("   Input data read, now calculating histograms\n");
	
	}
	
	int myrows = 100000 / world_size;
	int mystart = rank * myrows;
	
	if ( rank == 0 && myrows*world_size == 100000) printf("Each rank has %d rows of random data\n", myrows);
	if ( rank == world_size-1 && myrows*world_size != 100000 )
	{
		printf("Each rank has %d rows of random data", myrows);
		myrows = myrows + (100000 - myrows*world_size);
		printf("except rank %d with %d rows\n", world_size-1,myrows);
	}
	
	//Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);
	printf("Starting on processor %s, rank %3d out of %d processors\n", processor_name, rank, world_size);
	
	//Send data to ranks
	MPI_Bcast(real_rasc,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(real_decl,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(rand_rasc,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	MPI_Bcast(rand_decl,100000,MPI_FLOAT,0,MPI_COMM_WORLD);
	
	//Initialize histograms
    long int *histogram_DD, *histogram_DR, *histogram_RR;
	histogram_DD = (long int *)calloc(360L,sizeof(long int));
	histogram_DR = (long int *)calloc(360L,sizeof(long int));
	histogram_RR = (long int *)calloc(360L,sizeof(long int));
    MemoryAllocatedCPU += 3L*360L*sizeof(long int);
	for ( int i = 0; i<360; ++i) 
	{
		histogram_DD[i]=0L;
		histogram_DR[i]=0L;
		histogram_RR[i]=0L;
	}
    
	//Calculate angles
	calc_angles_RR(myrows, mystart, rand_decl, rand_rasc, histogram_RR);
	calc_angles_DR(myrows, mystart, real_decl, real_rasc, rand_decl, rand_rasc, histogram_DR);
	calc_angles_DD(myrows, mystart, real_decl, real_rasc, histogram_DD);
	
	//Allocate final histograms
	long int *final_histogram_DD, *final_histogram_RR, *final_histogram_DR;
	if ( rank == 0 )
	{
		final_histogram_DD = (long int *)calloc(360L, sizeof(long int));
		final_histogram_DR = (long int *)calloc(360L, sizeof(long int));
		final_histogram_RR = (long int *)calloc(360L, sizeof(long int));
		MemoryAllocatedCPU += 3L*360L*sizeof(long int);
		for ( int i = 0; i<360; ++i)
		{
			final_histogram_DD[i]=0L;
			final_histogram_DR[i]=0L;
			final_histogram_RR[i]=0L;
		}
	}
	
	MPI_Reduce(histogram_DD, final_histogram_DD, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(histogram_DR, final_histogram_DR, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(histogram_RR, final_histogram_RR, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if ( rank == 0 )
	{
		long int histsum = 0L;
		int correct_value=1;
		for ( int i = 0; i < 360; ++i ) histsum += final_histogram_DD[i];
			printf("   Histogram DD : sum = %ld\n",histsum);
				if ( histsum != 10000000000L ) correct_value = 0;

		histsum = 0L;
		for ( int i = 0; i < 360; ++i ) histsum += final_histogram_DR[i];
			printf("   Histogram DR : sum = %ld\n",histsum);
			if ( histsum != 10000000000L ) correct_value = 0;

		histsum = 0L;
		for ( int i = 0; i < 360; ++i ) histsum += final_histogram_RR[i];
			printf("   Histogram RR : sum = %ld\n",histsum);
			if ( histsum != 10000000000L ) correct_value = 0;

		if ( correct_value != 1 ) 
		{
		   printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);
		}
		
		printf("   Omega values for the histograms:\n");
		float omega[360];
		for ( int i = 0; i < 10; ++i ) 
			if ( final_histogram_RR[i] != 0L )
           {
				omega[i] = (final_histogram_DD[i] - 2L*final_histogram_DR[i] + final_histogram_RR[i])/((float)(final_histogram_RR[i]));
				if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i*0.25, (i+1)*0.25, omega[i]);
           }

		FILE *out_file = fopen(argv[3],"w");
		if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
		else
		{
			for ( int i = 0; i < 360; ++i ) 
				if ( final_histogram_RR[i] != 0L ) fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] ); 
			fclose(out_file);
		printf("   Omega values written to file %s\n",argv[3]);
       }
	}
	
    free(real_rasc); free(real_decl);
    free(rand_rasc); free(rand_decl);
	free(histogram_DD); free(histogram_DR); free(histogram_RR);
	
	if ( rank == 0 ) 
	{
		printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
		gettimeofday(&_ttime, &_tzone);
		double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
		printf("   Wall clock run time    = %.1lf secs\n",time_end - time_start);
	}
	MPI_Finalize();
    return(0);
}

int parseargs_readinput(int argc, char *argv[])
    {
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;
  
    if ( argc != 4 ) 
       {
       printf("   Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
       return(1);
       }
    if ( argc == 4 )
       {
       printf("   Running galaxy_openmp %s %s %s\n",argv[1], argv[2], argv[3]);

       real_data_file = fopen(argv[1],"r");
       if ( real_data_file == NULL ) 
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open real data file %s\n",argv[1]);
          return(1);
          }
       else
	  {
          fscanf(real_data_file,"%d",&Number_of_Galaxies);
          for ( int i = 0; i < 100000; ++i ) 
              {
      	      float rasc, decl;
	      if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[1]);
                 fclose(real_data_file);
	         return(1);
	         }
	      real_rasc[i] = rasc*arcmin2rad;
	      real_decl[i] = decl*arcmin2rad;
	      }
           fclose(real_data_file);
	   printf("   Successfully read 100000 lines from %s\n",argv[1]);
	   }

       rand_data_file = fopen(argv[2],"r");
       if ( rand_data_file == NULL ) 
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open random data file %s\n",argv[2]);
          return(1);
          }
       else 
	  {
          fscanf(rand_data_file,"%d",&Number_of_Galaxies);
          for ( int i = 0; i < 100000; ++i ) 
              {
      	      float rasc, decl;
	      if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[2]);
                 fclose(rand_data_file);
	         return(1);
	         }
	      rand_rasc[i] = rasc*arcmin2rad;
	      rand_decl[i] = decl*arcmin2rad;
	      }
          fclose(rand_data_file);
	  printf("   Successfully read 100000 lines from %s\n",argv[2]);
	  }
       out_file = fopen(argv[3],"w");
       if ( out_file == NULL ) 
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open output file %s\n",argv[3]);
          return(1);
          }
       else fclose(out_file);
       }

    return(0);
    }
	
void calc_angles_DD(int rows, int start, float *real_d, float *real_r, long int *hist)
{	
	for ( int i = start; i < start+rows; ++i )
	{
		float x1 = real_d[i];
		float x2 = real_r[i];
		for ( int j = 0; j < 100000; ++j )
		{
			float y1 = real_d[j];
			float y2 = real_r[j];
			float tmp = x2 - y2;
			float temp = sinf(x1) * sinf(y1) + cosf(x1) * cosf(y1) * cosf(tmp);
			if ( temp > 1.0f ) temp = 1.0f;
			//if ( temp < -1.0f ) temp = -1.0f;
			float angle = acosf(temp);
			angle = angle / pif * 180.0f;
			hist[(int)(4.0f * angle)] += 1L; 
		}
	}
}
	
void calc_angles_RR(int rows, int start, float *rand_d, float *rand_r, long int *hist) 
{
	for ( int i = start; i < start+rows; ++i )
	{
		float x1 = rand_d[i];
		float x2 = rand_r[i];
		for ( int j = 0; j < 100000; ++j )
		{
			float y1 = rand_d[j];
			float y2 = rand_r[j];
			float tmp = x2 - y2;
			float temp = sinf(x1) * sinf(y1) + cosf(x1) * cosf(y1) * cosf(tmp);
			if ( temp > 1.0f ) temp = 1.0f;
			//if ( temp < -1.0f ) temp = -1.0f;
			float angle = acosf(temp);
			angle = angle / pif * 180.0f;
			hist[(int)(4.0f * angle)] += 1L; 
		}
	}
}

void calc_angles_DR(int rows, int start, float *real_d, float *real_r, float *rand_d, float *rand_r, long int *hist)	
{
	for ( int i = start; i < start+rows; ++i )
	{
		float x1 = real_d[i];
		float x2 = real_r[i];
		for ( int j = 0; j < 100000; ++j )
		{
			float y1 = rand_d[j];
			float y2 = rand_r[j];
			float tmp = x2 - y2;
			float temp = sinf(x1) * sinf(y1) + cosf(x1) * cosf(y1) * cosf(tmp);
			if ( temp > 1.0f ) temp = 1.0f;
			float angle = acosf(temp);
			angle = angle / pif * 180.0f;
			hist[(int)(4.0f * angle)] += 1L; 
		}
	}
}
	