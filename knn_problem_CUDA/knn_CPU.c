#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
int N = 0, M = 0, K = 0;
typedef struct {
    int id;
    double dis;
}matrix;

matrix **matdis;
int **inmat;

int cmp (const void *a, const void *b) {
    matrix *c = (matrix *)a;
    matrix *d = (matrix *)b;
    if (c->dis < d->dis) return 0;
    else if (c->dis > d->dis) return 1;
    else return c->id > d->id; 
}

double distance(int *a, int *b) {
    int i;
    double d = 0.0;
    for (i=0; i<N; i++)
        d += (a[i]-b[i])*(a[i]-b[i]);

    return sqrt(d);
}

void knnCPU () {
    int i, j;
    for (i=0; i<M; i++)
        for (j=0; j<M; j++) {
            matdis[i][j].id = j;
            matdis[i][j].dis = 0;
        }

    for (i=0; i<M; i++)
        for (j=i+1; j<M; j++)
            matdis[i][j].dis = matdis[j][i].dis = distance(inmat[i], inmat[j]);

    for (i=0; i<M; i++)
        qsort(matdis[i], M, sizeof(matdis[0][0]), cmp);
}

int main (int argc, char **argv) {
    if (argc!=2) {
        printf("usage: ./a.out <testcase>\n");
        return -1;
    }	
    int i, j;
    FILE *tc, *out;
    tc = fopen(argv[1], "r");
    out = fopen("output_knn_CPU.txt", "w");
    fscanf(tc, "%d %d %d", &M, &N, &K);

    inmat = (int**)malloc(sizeof(int*)*M);
    for (i=0; i<M; i++)
        inmat[i] = (int*)malloc(sizeof(int)*N);

    matdis = (matrix**)malloc(sizeof(matrix*)*M);
    for (i=0; i<M; i++)
        matdis[i] = (matrix*)malloc(sizeof(matrix)*M);

    for (i=0; i<M; i++)
        for (j=0; j<N; j++)
            fscanf(tc, "%d", &inmat[i][j]);

    knnCPU();
    for (i=0; i<M; i++)
        for (j=1; j<=K; j++)
            fprintf(out, "%d ", matdis[i][j].id);
        fprintf(out, "\n");

    fclose(tc);
    fclose(out);
    return 0;
}
