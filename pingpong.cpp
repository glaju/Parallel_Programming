#include <random>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <math.h>

using namespace std;

int main(int argc, char *argv[])
{



    MPI_Init(&argc,&argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);



    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);



std::chrono::time_point<std::chrono::system_clock> start, end;
int n = pow(2,29);
char *buffer= new char[n];
char *rbuffer = new char[n];
    if(rank == 0)
    {
        for(int i=4; i<=pow(2,29); i*=2)
        {

            start = std::chrono::system_clock::now();

            MPI_Send(buffer,i,MPI_CHAR,1,i,MPI_COMM_WORLD);


            MPI_Status status;
            MPI_Recv(rbuffer,i,MPI_CHAR,1,i,MPI_COMM_WORLD,&status);
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            std::cout << "Runtime for " <<i<< " bytes: "<< elapsed_seconds.count() << "s\n";

        }

    }
    else
    {
        for(int i=4; i<=pow(2,29); i*=2){

        MPI_Status status;
        MPI_Recv(rbuffer,i,MPI_CHAR,0,i,MPI_COMM_WORLD,&status);


        MPI_Send(buffer,i,MPI_CHAR,0,i,MPI_COMM_WORLD);
        }


    }

    MPI_Finalize();

}


