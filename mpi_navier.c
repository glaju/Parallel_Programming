#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3
int my_rank;
int my_rank_c;
int nprocs;
int nprocs_y;
int nprocs_x;
int my_rank_x;
int my_rank_y;
int prev_y;
int next_y;
int next_x;
int prev_x;
MPI_Datatype vertSlice, horizSlice;
int imax_full;
int jmax_full;
int gbl_i_begin;
int gbl_j_begin;

double* dat_ptrs[6];
int dat_dirty[6] = {1,1,1,1,1,1};

void mpi_setup(int argc, char **argv, int *imax, int *jmax) {
	//Initialise: get #of processes and process id
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  int sides[2]={0,0};
  MPI_Dims_create(nprocs,2,sides);

    if (sides[0] != sides[1])
    {
        printf("Error, requires a square number of processes");
        MPI_Abort(MPI_COMM_WORLD,-1);
        exit(1);
    }
  MPI_Comm cart_comm;
  int dims[2] = {sides[0],sides[1]};
  int periods[2] = {1,1};
  int reorder = 0;
  int coords[2];
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
  MPI_Comm_rank(cart_comm,&my_rank_c);
  MPI_Cart_coords(cart_comm, my_rank_c, 2, coords);
  int idx_row = coords[0];
  int idx_col = coords[1];
  int neighbours[4];
  MPI_Cart_shift(cart_comm,0,1,&neighbours[UP],&neighbours[DOWN]);
  MPI_Cart_shift(cart_comm,1,1,&neighbours[LEFT],&neighbours[RIGHT]);


	//Figure out process X,Y coordinates
	nprocs_x = sqrt(nprocs);
	nprocs_y = sqrt(nprocs);

  my_rank_x = my_rank % nprocs_x;
  my_rank_y = my_rank / nprocs_x;
//my_rank = my_rank_y*nprocs_x+my_rank_x;

	//Figure out neighbours
  prev_x = (my_rank_x-1)<0 ? MPI_PROC_NULL : my_rank-1;
  next_x = (my_rank_x+1)>=nprocs_x ? MPI_PROC_NULL : my_rank+1;

  prev_y = (my_rank_y-1)<0 ? MPI_PROC_NULL : my_rank-nprocs_x;
  next_y = (my_rank_y+1)>=nprocs_y ? MPI_PROC_NULL : my_rank+nprocs_x;

	//Save original full sizes in x and y directions
  imax_full = *imax;
  jmax_full = *jmax;

	//Modify imax and jmax (pay attention to integer divisions's rounding issues!)
	 *imax = (my_rank_x != nprocs_x-1) ? imax_full/nprocs_x : imax_full - my_rank_x * (imax_full/nprocs_x);
	 *jmax = (my_rank_y != nprocs_y-1) ? jmax_full/nprocs_y : jmax_full - my_rank_y * (jmax_full/nprocs_y);

	//Figure out beginning i and j index in terms of global indexing
  gbl_i_begin = my_rank_x * (imax_full/nprocs_x);
  gbl_j_begin = my_rank_y * (jmax_full/nprocs_y);

	//Let's set up MPI Datatypes
  //Homework: ghost cells are not 1 on each side, but 2! Change these to send 2 rows/columns at the same time

  MPI_Type_vector((*jmax)+4,2,(*imax)+4, MPI_DOUBLE, &vertSlice);

  //MPI_Type_vector((*imax)+4,2,1, MPI_DOUBLE, &horizSlice);
  MPI_Type_vector(2*((*imax)+4),1,1, MPI_DOUBLE, &horizSlice);
  MPI_Type_commit(&vertSlice);
  MPI_Type_commit(&horizSlice);

}

void exchange_halo(int imax, int jmax, double *arr) {
	int dirty = -1;
	for (int i = 0; i < 6; i++) {
		if ((double*)arr == dat_ptrs[i]) {
			if (dat_dirty[i]) dirty = i;
			break;
		}
	}
	if (dirty!=-1) {
    //Homework: ghost cells are not 1 on each side, but 2!
    // since we are sending 2 rows/columns, make sure the offsets into arr are right!
		//Exchange halos: top, bottom, left, right
		//jobbról küld, balra kap
		MPI_Sendrecv(&arr[0*(imax+4)+imax]     ,1,vertSlice,neighbours[RIGHT] ,0,
                 &arr[0*(imax+4)+0]        ,1,vertSlice,neighbours[LEFT],0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           /*
    MPI_Sendrecv(&arr[0*(imax+4)+imax]     ,1,vertSlice,next_x ,0,
                 &arr[0*(imax+4)+0]        ,1,vertSlice,prev_x,0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);*/


    //balról küld, jobbra kap
      MPI_Sendrecv(&arr[0*(imax+4)+2]     ,1,vertSlice,neighbours[LEFT] ,0,
                 &arr[0*(imax+4)+imax+2],1,vertSlice,neighbours[RIGHT],0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           /*
    MPI_Sendrecv(&arr[0*(imax+4)+2]     ,1,vertSlice,prev_x ,0,
                 &arr[0*(imax+4)+imax+2],1,vertSlice,next_x,0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);*/

    //alulról küld, felülre kap
    MPI_Sendrecv(&arr[(jmax)*(imax+4)+0] ,1,horizSlice,neighbours[DOWN],0,
                 &arr[0*(imax+4)+0]        ,1,horizSlice,neighbours[UP],0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           /*
    MPI_Sendrecv(&arr[(jmax)*(imax+4)+0] ,1,horizSlice,next_y,0,
                 &arr[0*(imax+4)+0]        ,1,horizSlice,prev_y,0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);*/

    //felülrõl küld, alulra kap
     MPI_Sendrecv(&arr[2*(imax+4)+0] ,1,horizSlice,neighbours[UP],0,
                 &arr[(jmax+2)*(imax+4)+0],1,horizSlice,neighbours[DOWN],0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);
           /*
    MPI_Sendrecv(&arr[2*(imax+4)+0] ,1,horizSlice,prev_y,0,
                 &arr[(jmax+2)*(imax+4)+0],1,horizSlice,next_y,0,
           MPI_COMM_WORLD,MPI_STATUS_IGNORE);*/

		dat_dirty[dirty] = 0;
	}
}

void set_dirty(double *arr) {
	for (int i = 0; i < 6; i++) {
		if ((double*)arr == dat_ptrs[i]) {
			dat_dirty[i] = 1;
			break;
		}
	}
}
