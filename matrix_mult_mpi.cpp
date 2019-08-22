#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>
#define N 40
using namespace std;

int A[N*N], B[N*N], C[N*N];
unsigned block_size;

void printMatrix(int matrix[], unsigned size) {
    for (unsigned k = 0; k < size; k++) {
        printf ("\t");
        for (unsigned l = 0; l < size; l++)
            printf ("%d\t", matrix[k*size + l]);
        printf ("\n");
    }
}

void send(void* buffer, int target) {
    MPI_Send(buffer,                 // buffer
             block_size*block_size,   // count of elements to send
             MPI_INT,                 // type of data
             target,                  // destination process
             22,                      // tag
             MPI_COMM_WORLD);         // communicator
}

void receive(void* buffer, int source) {
    MPI_Status status;
    MPI_Recv(buffer,                 // buffer
             block_size*block_size,   // count of elements to send
             MPI_INT,                 // type of data
             source,                  // source process
             22,                      // tag
             MPI_COMM_WORLD,          // communicator
             &status);                // status
}

// Copy values to submatrix from original large matrix
void copyToSubmatrix(int source[], unsigned position, int destination[]) {
    for (unsigned k = 0; k < block_size; k++)
        copy_n(source + position + k*N, block_size, destination + k*block_size);
}

// Copy values from submatrix back to the original large matrix
void copyFromSubmatrix(int source[], unsigned position, int destination[]) {
    for (unsigned k = 0; k < block_size; k++)
        copy_n(source + k*block_size, block_size, destination + position + k*N);
}

int main(int argc, char *argv[]) {
    int  numtasks, rank, len, rc;
    double start, end;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    rc = MPI_Init(&argc,&argv);
    if (rc != MPI_SUCCESS) {
        printf ("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned sqrt_numtasks = ceil(sqrt(numtasks));
    block_size = N/sqrt_numtasks;
    int blockA[block_size*block_size],
        blockB[block_size*block_size],
        blockC[block_size*block_size];

    // -------------- Initialization -----------------
    if(rank == 0) {
        // Checks on number of tasks
        if (sqrt_numtasks*sqrt_numtasks != numtasks)
            printf ("Warning: Number of processes should be square number. Only %d processes used.\n",
                    sqrt_numtasks*sqrt_numtasks);
        if (N % sqrt_numtasks != 0) {
            printf ("Error: Square root of number of processes should be divisor of N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize matrices
        srand(time(0));
        for (unsigned i = 0; i < N*N; i++) {
            B[i] = A[i] = i;
            //A[i] = rand() % 1e4;
            //B[i] = rand() % 1e4;
        }

        start = MPI_Wtime();

        // Split matrices into 'numtasks' pieces
        for (unsigned i = 0; i < N; i += block_size){
            for (unsigned j = 0; j < N; j += block_size) {
                if (i == 0 && j == 0) continue; // This is the master task's block!

                // Slice block from matrices
                copyToSubmatrix(A, i*N + j, blockA);
                copyToSubmatrix(B, i*N + j, blockB);


                // Sending block from matrix A
                unsigned I = i / block_size;
                unsigned J = ((j / block_size - I) + sqrt_numtasks) % sqrt_numtasks;
                send(&blockA, I * sqrt_numtasks + J);

                // Sending block from matrix B
                J = j / block_size;
                I = int(((i / block_size - J) + sqrt_numtasks)) % sqrt_numtasks;
                send(&blockB, I * sqrt_numtasks + J);

            }
        }

        // A_0,0 and B_0,0 is the master task's blocks
        copyToSubmatrix(A, 0, blockA);
        copyToSubmatrix(B, 0, blockB);

    } else if (rank < sqrt_numtasks*sqrt_numtasks) {
        receive(&blockA, 0);
        receive(&blockB, 0);

    } else { // I cannot use these processes because number of processes have to be square number
        MPI_Finalize();
        return 0;
    }

    // -------------- Calculation -----------------

    // Initial multiplication
    for (unsigned i = 0; i < block_size; i++){
        for (unsigned j = 0; j < block_size; j++) {
            blockC[i*block_size + j] = 0;
            for (unsigned k = 0; k < block_size; k++) {
                blockC[i*block_size + j] += blockA[i*block_size + k] * blockB[k*block_size + j];
            }
        }
    }

    // Further steps
    unsigned rank_i = rank / sqrt_numtasks;
    unsigned rank_j = rank % sqrt_numtasks;
    for (unsigned step = 1; step < sqrt_numtasks; step++) {
        // Shift blocks
        send(&blockA, rank_i * sqrt_numtasks + (rank_j - 1 + sqrt_numtasks) % sqrt_numtasks);
        send(&blockB, ((rank_i - 1 + sqrt_numtasks) % sqrt_numtasks) * sqrt_numtasks + rank_j);
        receive(&blockA, rank_i * sqrt_numtasks + (rank_j + 1 + sqrt_numtasks) % sqrt_numtasks);
        receive(&blockB, ((rank_i + 1 + sqrt_numtasks) % sqrt_numtasks) * sqrt_numtasks + rank_j);

        // Do multiplication
        for (unsigned i = 0; i < block_size; i++){
            for (unsigned j = 0; j < block_size; j++) {
                for (unsigned k = 0; k < block_size; k++) {
                    blockC[i*block_size + j] += blockA[i*block_size + k] * blockB[k*block_size + j];
                }
            }
        }
    }

    // -------------- Termination -----------------
    if (rank == 0) {
        // That block was calculated by master process
        copyFromSubmatrix(blockC, 0, C);

        // Collect results from other processes
        for (unsigned i = 0; i < sqrt_numtasks; i++){
            for (unsigned j = 0; j < sqrt_numtasks; j++) {
                if (i == 0 && j == 0) continue;
                // Receive result
                receive(&blockC, i*sqrt_numtasks + j);
                // Copy to the proper place in matrix C
                copyFromSubmatrix(blockC, i*block_size*block_size + j*block_size, C);
            }
        }

        end = MPI_Wtime();
        printMatrix(C, N);
        printf("Runtime = %f\n", end-start);
    } else {
        send(&blockC, 0);
    }

    MPI_Finalize();

}
