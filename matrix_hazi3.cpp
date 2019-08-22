#include <random>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <mpi.h>


using namespace std;



int main(int argc, char *argv[])
{

    int N=20;
    int chunk_size;
    int a[N*N];
    int b[N*N];
    int c[N*N];


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 100.0);




    MPI_Init(&argc,&argv);
    int p;
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    chunk_size= N/sqrt(p);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int part_a[chunk_size*chunk_size];
    int part_b[chunk_size*chunk_size];
    int part_c[chunk_size*chunk_size];
    MPI_Status status;
    if(rank == 0)
    {

        for(int i=0; i<N*N; i++)
        {
            a[i] = dis(gen);
        }
        for(int i=0; i<N*N; i++)
        {
            b[i] = dis(gen) ;
        }



        for (int i = 0; i < N; i += chunk_size)
        {
            for (int j = 0; j < N; j += chunk_size)
            {
                if (i == 0 && j == 0)
                {
                    continue;
                }



                for (int k = 0; k < chunk_size; k++)
                {
                    copy_n(a + i*N + j + k*N, chunk_size, part_a + k*chunk_size);
                }

                for (int k = 0; k < chunk_size; k++)
                {
                    copy_n(b + i*N + j  + k*N, chunk_size,part_b + k*chunk_size);
                }

                int I = i / chunk_size;
                int J = int(((j / chunk_size - I) + sqrt(p))) % int(sqrt(p));
                MPI_Send(&part_a,chunk_size*chunk_size,MPI_INT, I * sqrt(p) + J,22,MPI_COMM_WORLD);

                J = j / chunk_size;
                I = int(((i / chunk_size - J) + sqrt(p))) % int(sqrt(p));
                MPI_Send(&part_b,chunk_size*chunk_size,MPI_INT, I * sqrt(p) + J,22,MPI_COMM_WORLD);


            }
        }



        for (int k = 0; k < chunk_size; k++)
        {
            copy_n(a  + k*N, chunk_size, part_a + k*chunk_size);
        }
        for (int k = 0; k < chunk_size; k++)
        {
            copy_n(b  + k*N, chunk_size, part_b + k*chunk_size);
        }


    }
    else
    {

        MPI_Recv(&part_a,chunk_size*chunk_size,MPI_INT,0,22,MPI_COMM_WORLD,&status);

        MPI_Recv(&part_b,chunk_size*chunk_size,MPI_INT,0,22,MPI_COMM_WORLD,&status);


    }
    for (int i = 0; i < chunk_size; i++)
    {
        for (int j = 0; j < chunk_size; j++)
        {
            for (int k = 0; k < chunk_size; k++)
            {
                part_c[i*chunk_size + j] = part_a[i*chunk_size + k] * part_b[k*chunk_size + j];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int sq_p = sqrt(p);
    int rank1 = rank / sq_p;
    int rank2 = rank % sq_p;
    for (int i = 1; i < sq_p; i++)
    {

        MPI_Send(&part_a,chunk_size*chunk_size,MPI_INT, rank1 * sq_p + (rank2 - 1 + sq_p) % sq_p,22,MPI_COMM_WORLD);
        MPI_Send(&part_b,chunk_size*chunk_size,MPI_INT,((rank1 - 1 + sq_p) % sq_p) * sq_p + rank2,22,MPI_COMM_WORLD);
        MPI_Recv(&part_a,chunk_size*chunk_size,MPI_INT,rank1 * sq_p + (rank2 + 1 + sq_p) % sq_p,22,MPI_COMM_WORLD,&status);
        MPI_Recv(&part_b,chunk_size*chunk_size,MPI_INT,((rank1 + 1 + sq_p) % sq_p) * sq_p + rank2,22,MPI_COMM_WORLD,&status);


        for (int l = 0; l < chunk_size; l++)
        {
            for (int j = 0; j < chunk_size; j++)
            {
                for (int k = 0; k < chunk_size; k++)
                {
                    part_c[l*chunk_size + j] += part_a[l*chunk_size + k] * part_b[k*chunk_size + j];
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0)
    {

        for (int i = 0; i < sq_p; i++)
        {
            for (int j = 0; j < sq_p; j++)
            {
                if (i == 0 && j == 0) continue;
                MPI_Recv(&part_c,chunk_size*chunk_size,MPI_INT,i*sq_p + j,22,MPI_COMM_WORLD,&status);



                for (int k = 0; k < chunk_size; k++)
                {
                    copy_n(part_c + k*chunk_size, chunk_size, c + i*chunk_size*chunk_size + j*chunk_size + k*N);
                }
            }
        }


        for (int k = 0; k < chunk_size; k++)
        {
            copy_n(part_c + k*chunk_size, chunk_size, c +  k*N);
        }


    }
    else
    {

        MPI_Send(&part_c,chunk_size*chunk_size,MPI_INT, 0,22,MPI_COMM_WORLD);

    }


    MPI_Finalize();

    return 0;
}
