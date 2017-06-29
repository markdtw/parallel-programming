#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#define GCONST	6.67384e-11
#define PI		3.1415926
#define WHITE	0xFFFFFF
#define BLACK	0x000000
typedef struct {
    double x, y;
    double vx, vy;
}planet;
typedef struct bhtree{
    int num_body;
    double cxn, cyn;
    planet *body;
    struct bhtree *nw;
    struct bhtree *ne;
    struct bhtree *sw;
    struct bhtree *se;
}bhtree;

planet *planets;
bhtree *tree;
Display *display;
Window win;
GC gc;

int en = 1, ptr = 0;
int STARS = 0, THREADS = 0, STEPS = 0, THETA = 0;
double MASS = 0.0, TIME = 0.0, CLEN = 0.0, WLEN = 0.0, XMIN = 0.0, YMIN = 0.0;
double maxx = 0.0, maxy = 0.0, minx = 0.0, miny = 0.0, disx = 0.0, disy = 0.0;
FILE *dev;

void computeV (planet *p, bhtree *node) {
    double disx, disy, dis;
    disx = node->cxn - p->x;
    disy = node->cyn - p->y;
    dis = sqrt(disx*disx + disy*disy);
    if (dis!=0) {
        p->vx -= GCONST * MASS/(dis*dis) * (disx/dis);
        p->vy -= GCONST * MASS/(dis*dis) * (disy/dis);
    }
}
void updateF (bhtree *node, planet *p, double s) {
    double d = sqrt((p->x-node->cxn/node->num_body) * (p->x-node->cxn/node->num_body) +
            (p->y-node->cyn/node->num_body) * (p->y-node->cyn/node->num_body));
    if (node->num_body==1)
        computeV(p, node);
    else if (s/d<THETA)
        computeV(p, node);
    else {
        if (node->nw!=NULL) updateF(node->nw, p, s/2);
        if (node->ne!=NULL) updateF(node->ne, p, s/2);
        if (node->sw!=NULL) updateF(node->sw, p, s/2);
        if (node->se!=NULL) updateF(node->se, p, s/2);
    }
}
void sequential () {
    int i;
    maxx = maxy = 0.0;
    minx = miny = 100.0;
    for (i=0; i<STARS; i++) {
        planet *p = &(planets[i]);
        updateF(tree, p, maxx-minx);
    }
    for (i=0; i<STARS; i++) {
        planet *p = &(planets[i]);
        p->x += TIME * p->vx;
        p->y += TIME * p->vy;
        if (p->x>=maxx) maxx = p->x;
        if (p->x<=minx) minx = p->x;
        if (p->y>=maxy) maxy = p->y;
        if (p->y<=miny) miny = p->y;
    }
}
bhtree *init () {
    bhtree *node = (bhtree*)malloc(sizeof(bhtree));
    node->num_body = 0;
    node->cxn = node->cyn = 0.0;
    node->body = NULL;
    node->nw = node->ne = node->sw = node->se = NULL;
    return node;
}
void Quad_insert (planet *p, bhtree *node, double px, double nx, double py, double ny) {
    double rangex = (px+nx)/2;
    double rangey = (py+ny)/2;

    node->cxn += p->x;
    node->cyn += p->y;

    if (node->body==NULL && node->nw==NULL && node->ne==NULL && node->sw==NULL && node->se==NULL && node->num_body==0) {
        node->body = p;
    } else if (node->body!=NULL &&node->nw==NULL &&node->ne==NULL &&node->sw==NULL &&node->se==NULL &&node->num_body==1){
        node->nw = init();
        node->ne = init();
        node->sw = init();
        node->se = init();
        if (node->body->x < rangex && node->body->y > rangey) {
            // north west
            Quad_insert(node->body, node->nw, rangex, nx, py, rangey);
        } else if (node->body->x >= rangex && node->body->y > rangey) {
            // north east
            Quad_insert(node->body, node->ne, px, rangex, py, rangey);
        } else if (node->body->x < rangex && node->body->y <= rangey) {
            // south west
            Quad_insert(node->body, node->sw, rangex, nx, rangey, ny);
        } else {
            // south east
            Quad_insert(node->body, node->se, px, rangex, rangey, ny);
        } if (p->x < rangex && p->y > rangey) {
            // north west
            Quad_insert(p, node->nw, rangex, nx, py, rangey);
        } else if (p->x >= rangex && p->y > rangey) {
            // north east
            Quad_insert(p, node->ne, px, rangex, py, rangey);
        } else if (p->x < rangex && p->y <= rangey) {
            // south west
            Quad_insert(p, node->sw, rangex, nx, rangey, ny);
        } else {
            // south east
            Quad_insert(p, node->se, px, rangex, rangey, ny);
        }
        node->body = NULL;
    } else {
        if (p->x < rangex && p->y > rangey) {
            // north west
            Quad_insert(p, node->nw, rangex, nx, py, rangey);
        } else if (p->x >= rangex && p->y > rangey) {
            // north east
            Quad_insert(p, node->ne, px, rangex, py, rangey);
        } else if (p->x < rangex && p->y <= rangey) {
            // south west
            Quad_insert(p, node->sw, rangex, nx, rangey, ny);
        } else {
            // south east
            Quad_insert(p, node->se, px, rangex, rangey, ny);
        }
    }
    node->num_body++;
}
void build_tree () {
    int i;
    tree = init();
    for (i=0; i<STARS; i++) {
        planet *p = &(planets[i]);
        Quad_insert(p, tree, maxx, minx, maxy, miny);
    }
}

void freetree (bhtree *node) {
    if (node!=NULL) {
        freetree(node->nw);
        freetree(node->ne);
        freetree(node->sw);
        freetree(node->se);
        node->body = NULL;
        free(node);
        node = NULL;
    }
}
int main (int argc, char **argv) {
    if (argc!=8 && argc!=12) {
        printf("Usage: ./a.out #threads m T t FILE θ enable/disable xmin ymin len Len\nExiting...\n");
        exit(1);
    }
    THREADS = atoi(argv[1]);
    MASS = atof(argv[2]);
    STEPS = atoi(argv[3]);
    TIME = atof(argv[4]);
    THETA = atoi(argv[6]);
    if (strcmp(argv[7], "enable")==0) {
        if (argc!=12) {
            printf("Miss 4 arguments.\nExiting...\n\n");
            exit(1);
        }
        XMIN = atof(argv[8]);
        YMIN = atof(argv[9]);
        CLEN = atof(argv[10]);
        WLEN = atof(argv[11]);
    } else if (strcmp(argv[7], "disable")==0) {
        en = 0;
        printf("Xwindow disabled.\n");
    } else {
        printf("please enter 'enable' or 'disable'.\nExiting...\n\n");
        exit(1);
    }

    //	planet *planets;
    FILE *file = fopen(argv[5], "r");
    dev  = fopen("d.txt", "w");
    if (file==0) {
        printf("Testcase missing.\nExiting...\n\n");
        exit(1);
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
        disx = maxx - minx;
        disy = maxy - miny;
        fclose(file);
    }

    /* Handle Xwindow */
    if (en) {
        int screen;
        display = XOpenDisplay(NULL);
        if(display == NULL) {
            printf("cannot open display\nExiting...\n\n");
            exit(1);
        }
        screen = DefaultScreen(display);
        win = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, WLEN, WLEN, 0, BLACK, BLACK);
        XMapWindow(display, win);
        XSync(display, 0);
        gc = XCreateGC(display, win, 0, 0);
        XSetForeground(display, gc, WHITE);
        XFlush(display);
    }

    /* Sequential */
    //	fprintf(dev, "tree = %d\n", tree->num_body);
    //	fprintf(dev, "tree: nw = %d, ne = %d, sw = %d, se = %d\n", tree->nw->num_body, tree->ne->num_body, tree->sw->num_body, tree->se->num_body);
    //	fprintf(dev, "treenw: nw = %d, ne = %d, sw = %d, se = %d\n", tree->nw->nw->num_body, tree->nw->ne->num_body, tree->nw->sw->num_body, tree->nw->se->num_body);
    //	fprintf(dev, "treene: nw = %d, ne = %d, sw = %d, se = %d\n", tree->ne->nw->num_body, tree->ne->ne->num_body, tree->ne->sw->num_body, tree->ne->se->num_body);
    //	fprintf(dev, "treene: nw cxn = %.3lf, cyn = %.3lf\n", tree->ne->nw->cxn, tree->ne->nw->cyn);
    //	fprintf(dev, "treenenw: nw = %d, ne = %d, sw = %d, se = %d\n", tree->ne->nw->nw->num_body, tree->ne->nw->ne->num_body, tree->ne->nw->sw->num_body, tree->ne->nw->se->num_body);
    //	fprintf(dev, "treesw: nw = %d, ne = %d, sw = %d, se = %d\n", tree->sw->nw->num_body, tree->sw->ne->num_body, tree->sw->sw->num_body, tree->sw->se->num_body);
    //	fprintf(dev, "root cxn = %.3lf, cyn = %.3lf\n", tree->cxn, tree->cyn);
    //	fprintf(dev, "treenw: cxn = %.3lf, cyn = %.3lf\n", tree->nw->cxn, tree->nw->cyn);
    //	fprintf(dev, "treenenw: cxn = %.3lf, cyn = %.3lf\n", tree->ne->nw->cxn, tree->ne->nw->cyn);

    struct timespec tstart, tstop;
    int i, j;
    clock_gettime(CLOCK_REALTIME, &tstart);
    for (i=0; i<STEPS; i++) {
        build_tree();
        sequential();
        freetree(tree);
        if (en) {
            XClearWindow(display, win);
            for (j=0; j<STARS; j++) {
                planet *p = &(planets[j]);
                XFillArc(display, win, gc, (p->x-XMIN)*WLEN/CLEN, WLEN-(p->y-YMIN)*WLEN/CLEN, 2, 2, 0, 360*64);
            }
            XFlush(display);
            usleep(5000);
        }
    }
    clock_gettime(CLOCK_REALTIME, &tstop);
    double sdur = 1000000*(tstop.tv_sec-tstart.tv_sec)+(tstop.tv_nsec-tstart.tv_nsec)/1000;
    printf("\nIt took %.5lf seconds to finish sequential caculation.\n", sdur/1000000);

    fclose(dev);
    return 0;
}
