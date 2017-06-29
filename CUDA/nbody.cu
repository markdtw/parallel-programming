#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define BLOCKSPERGRID 512
#define GCONST	6.67384e-11
#define PI	3.1415926
#define WHITE	0xFFFFFF
#define BLACK	0x000000
typedef struct {
    double x, y;
    double vx, vy;
}planet;

Display *display;
Window win;
GC gc;

int THREADSPERBLOCK = 0;
int en = 1, ptr = 0;
int STARS = 0, THREADS = 0, STEPS = 0, THETA = 0;
double MASS = 0.0, TIME = 0.0, CLEN = 0.0, WLEN = 0.0, XMIN = 0.0, YMIN = 0.0;
double maxx = 0.0, maxy = 0.0, minx = 0.0, miny = 0.0;
FILE *dev;

__global__ void n2simulation (planet *p, int stars, double time, double mass, double g) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j;

    planet *p1 = &(p[i]);
    for (j=0; j<stars; j++) {
        planet *p2 = &(p[j]);
        double disx, disy, dis;
        disx = p2->x - p1->x;
        disy = p2->y - p1->y;
        dis = sqrt(disx*disx + disy*disy);
        if (dis!=0) {
            p1->vx -= g * mass / (dis*dis) * (disx/dis) * time;
            p1->vy -= g * mass / (dis*dis) * (disy/dis) * time;
        }
    }
    p1->x += time * p1->vx;
    p1->y += time * p1->vy;
}

int main (int argc, char **argv) {
    if (argc!=8 && argc!=12) {
        printf("usage: ./a.out #threads m T t FILE θ enable/disable xmin ymin len Len\n");
        return -1;
    }

    THREADS = atoi(argv[1]);
    MASS = atof(argv[2]);
    STEPS = atoi(argv[3]);
    TIME = atof(argv[4]);
    THETA = atoi(argv[6]);
    if (strcmp(argv[7], "enable")==0) {
        if (argc!=12) {
            printf("Miss 4 arguments.\n");
            return -1;
        }
        XMIN = atof(argv[8]);
        YMIN = atof(argv[9]);
        CLEN = atof(argv[10]);
        WLEN = atof(argv[11]);
    } else if (strcmp(argv[7], "disable")==0) {
        en = 0;
        printf("Xwindow disabled.\n");
    } else {
        printf("please enter 'enable' or 'disable'.\n");
        return -1;
    }

    //	planet *planets;
    FILE *file = fopen(argv[5], "r");
    dev  = fopen("d.txt", "w");
    planet *planets;
    if (file==0) {
        printf("Testcase missing.\n");
        return -1;
    } else {
        fscanf(file, "%d", &STARS);
        planets = (planet*)malloc(sizeof(planet)*STARS);
        int i;
        for (i=0; i<STARS; i++) {
            fscanf(file, "%lf %lf %lf %lf", &planets[i].x, &planets[i].y, &planets[i].vx, &planets[i].vy);
            if (planets[i].x>maxx)
                maxx = planets[i].x;
            if (planets[i].x<minx)
                minx = planets[i].x;
            if (planets[i].y>maxy)
                maxy = planets[i].y;
            if (planets[i].y<miny)
                miny = planets[i].y;
        }
        fclose(file);
    }

    /* Handle Xwindow */
    if (en) {
        int screen;
        display = XOpenDisplay(NULL);
        if(display == NULL) {
            printf("cannot open display.\n");
            return -1;
        }
        screen = DefaultScreen(display);
        win = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, WLEN, WLEN, 0, BLACK, BLACK);
        XMapWindow(display, win);
        XSync(display, 0);
        XMoveWindow(display, win, 0, 0);
        gc = XCreateGC(display, win, 0, 0);
        XSetForeground(display, gc, WHITE);
        XFlush(display);
    }

    planet *d_p;
    int d_stars = STARS;
    double d_time = TIME, d_mass = MASS, d_g = GCONST;
    cudaError_t R;
    R = cudaMalloc((void **)&d_p, sizeof(planet)*STARS);
    R = cudaMemcpy(d_p, planets, sizeof(planet)*STARS, cudaMemcpyHostToDevice);

    THREADSPERBLOCK = STARS/BLOCKSPERGRID+1;

    int i, j;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    struct timespec cstart, cstop;
    double cdur=0.0, ct=0.0;

    for (i=0; i<STEPS; i++) {
        clock_gettime(CLOCK_REALTIME, &cstart);
        n2simulation<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_p, d_stars, d_time, d_mass, d_g);
        cudaThreadSynchronize();
        clock_gettime(CLOCK_REALTIME, &cstop);
        cdur = 1000000*(cstop.tv_sec-cstart.tv_sec)+(cstop.tv_nsec-cstart.tv_nsec)/1000;
        fprintf(dev, "It took %.7lf seconds to compute.\n", cdur/1000000);
        ct+=cdur/1000000;

        R = cudaMemcpy(planets, d_p, sizeof(planet)*STARS, cudaMemcpyDeviceToHost);
        if (en) {
            XClearWindow(display, win);
            for (i=0; i<STARS; i++) {
                double posx = (planets[i].x-XMIN)*WLEN/CLEN, posy = WLEN-(planets[i].y-YMIN)*WLEN/CLEN;
                XFillArc(display, win, gc, posx, posy, 2, 2, 0, 360*64);
            }
            XFlush(display);
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    fprintf(dev, "It took %.7lf seconds to compute. (Average)\n", ct/STEPS);
    fprintf(dev, "Execution Time on GPU: %3.20f s\n", elapsedTime/1000);

    cudaFree(d_p);

    fclose(dev);
    return 0;
}
