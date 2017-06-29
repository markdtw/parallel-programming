#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <pthread.h>

#define NUM_THREADS 5

#define WHITE 		0xFFFFFF
#define BLACK 		0x000000
#define LIPS_RED	0xCC0000
#define FROG_GREEN	0x006600
#define WOOD_BROWN	0x663300
#define RIVER_BLUE	0x0099FF
#define LIGHT_GREEN	0x66FF66

pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

int wood_num[4] = {0};
int wood_x[5] = {0, 0, 0, 300, 0};
int frog_x = 245;
int frog_y = 500;
int frog_cx;
int level = -1;
int halt = 0;
unsigned int slow   = 45000;
unsigned int medium = 30000;
unsigned int fast   = 15000;
unsigned int light  = 5000;
Display *display;
Window win;
GC river;
GC tree;
GC frog;

void drawall(int x_mv, int y_mv) {

    if (frog_x + x_mv >= 0 && frog_x + x_mv <= 500) 
        frog_x += x_mv;
    if (frog_y + y_mv >= 0 && frog_y + y_mv < 600)
        frog_y += y_mv;

    frog_cx = frog_x + 47;

    /* Set up river */
    XSetForeground(display, river, RIVER_BLUE);
    XFillRectangle(display, win, river, 0, 100, 600, 400);

    /* Set up tree (s) */
    XSetForeground(display, tree, WOOD_BROWN);
    XFillRectangle(display, win, tree, wood_x[0], 450, 100, 50);
    XFillRectangle(display, win, tree, wood_x[0]+200, 450, 100, 50);
    XFillRectangle(display, win, tree, wood_x[0]+400, 450, 100, 50);

    XFillRectangle(display, win, tree, wood_x[1], 350, 100, 50);
    XFillRectangle(display, win, tree, wood_x[1]+200, 350, 100, 50);

    XFillRectangle(display, win, tree, wood_x[2], 250, 100, 50);
    XFillRectangle(display, win, tree, wood_x[3], 250, 100, 50);

    XFillRectangle(display, win, tree, wood_x[4], 150, 150, 50);

    /* Set up frog */
    XSetForeground(display, frog, FROG_GREEN);
    XFillArc(display, win, frog, frog_x+5, frog_y+10, 80, 80, 0, 360*64); // Frog's body
    XFillArc(display, win, frog, frog_x, frog_y, 40, 40, 0, 360*64); // Frog's left eye
    XFillArc(display, win, frog, frog_x+45, frog_y, 40, 40, 0, 360*64); // Frog's right eye

    XSetForeground(display, frog, LIPS_RED);
    XFillArc(display, win, frog, frog_x+32, frog_y+50, 40, 40, 0, 360*64); // Frog's lips

    XSetForeground(display, frog, WHITE);
    XFillArc(display, win, frog, frog_x+2, frog_y+2, 35, 35, 0, 360*64); // Frog's left white eye
    XFillArc(display, win, frog, frog_x+47, frog_y+2, 35, 35, 0, 360*64); // Frog's right white eye
    XFillArc(display, win, frog, frog_x+45, frog_y+60, 20, 20, 0, 360*64); // Frog's mouth

    XSetForeground(display, frog, BLACK);
    XFillArc(display, win, frog, frog_x+5, frog_y+20, 25, 10, 0, 360*64); // Frog's left eyeball
    XFillArc(display, win, frog, frog_x+50, frog_y+20, 25, 10, 0, 360*64); // Frog's right eyeball

    XFlush(display);
}
void *float_wood (void *t) {
    int i, j;
    long tid = (long) t;
    int turn1=1, turn2=1, turn3_1=1, turn3_2=1, turn4=1;
    int move_wood=0;

    while (halt==0) {
        usleep(medium);
        pthread_mutex_lock(&count_mutex);

        XClearWindow(display, win);

        if (level==-1)
            drawall(0, 0);
        if (move_frog()==1)
            drawall(wood_x[0]-frog_x, 0);
        else if (move_frog()==2)
            drawall(wood_x[0]-frog_x+200, 0);
        else if (move_frog()==3)
            drawall(wood_x[0]-frog_x+400, 0);
        else if (move_frog()==4)
            drawall(wood_x[1]-frog_x, 0);
        else if (move_frog()==5)
            drawall(wood_x[1]-frog_x+200, 0);
        else if (move_frog()==6)
            drawall(wood_x[2]-frog_x, 0);
        else if (move_frog()==7)
            drawall(wood_x[3]-frog_x, 0);
        else if (move_frog()==8)
            drawall(wood_x[4]-frog_x+40, 0);
        else if (move_frog()==9)
            drawall(0, -100);

        if (tid==1) {
            if (turn1==1)
                wood_x[0] += 1;
            else
                wood_x[0] -= 1;
        } else if (tid==2) {
            if (turn2==1)
                wood_x[1] -= 4;
            else
                wood_x[1] += 4; 
        } else if (tid==3) {
            if (turn3_1==1)
                wood_x[2] += 5;
            else
                wood_x[2] -= 5;
            if (turn3_2==1)
                wood_x[3] -= 5;
            else
                wood_x[3] += 5;
        } else if (tid==4) {
            if (turn4==1)
                wood_x[4] -= 7;
            else
                wood_x[4] += 7;
        }
        if (wood_x[0]>=100)
            turn1 = 0;
        if (wood_x[0]<=0)
            turn1 = 1;

        if (wood_x[1]>=300)
            turn2 = 1;
        if (wood_x[1]<=0)
            turn2 = 0;

        if (wood_x[2]>=wood_x[3]-100)
            turn3_1 = 0;
        if (wood_x[2]<=0)
            turn3_1 = 1;
        if (wood_x[3]<=wood_x[2]+100)
            turn3_2 = 0;
        if (wood_x[3]>=500)
            turn3_2 = 1;

        if (wood_x[4]>=450)
            turn4 = 1;
        if (wood_x[4]<=0)
            turn4 = 0;

        pthread_mutex_unlock(&count_mutex);
    }
}
int move_frog () {
    if (level==0) {
        if (frog_cx>=wood_x[0] && frog_cx<=wood_x[0]+100)
            return 1;
        else if (frog_cx>=wood_x[0]+200 && frog_cx<=wood_x[0]+300)
            return 2;
        else if (frog_cx>=wood_x[0]+400 & frog_cx<=wood_x[0]+500)
            return 3;
    } else if (level==1) {
        if (frog_cx>=wood_x[1] && frog_cx<=wood_x[1]+100)
            return 4;
        else if (frog_cx>=wood_x[1]+200 && frog_cx<=wood_x[1]+300)
            return 5;
    } else if (level==2) {
        if (frog_cx>=wood_x[2] && frog_cx<=wood_x[2]+100)
            return 6;
        if (frog_cx>=wood_x[3] && frog_cx<=wood_x[3]+100)
            return 7;
    } else if (level==3) {
        if (frog_cx>=wood_x[4] && frog_cx<=wood_x[4]+100)
            return 8;
    } else if (level==4) {
        if (frog_cx>=0 && frog_cx<=600)
            return 9;
    }
    return 0;
}
void *draw_kb (void *t) {

    /* Set up Keyboard event */
    XEvent ev;
    /* Tell the display server what kind of events we would like to see */
    XSelectInput(display, win, ExposureMask | StructureNotifyMask | KeyPressMask);
    /* As each event that we asked about occurs, we respond. In this case we note if the window's
     * shape changed, and exit if a button is pressed inside the window.
     */

    int up = 0;

    while (halt==0) {
        XNextEvent(display, &ev);
        switch(ev.type) {
            case KeyPress: {
                char buf[25] = {0};
                int len;
                KeySym keysym;
                len = XLookupString(&ev.xkey, buf, 25, &keysym, NULL);
                switch (keysym) {
                    case XK_Left:
                        if (up==0) {
                            XClearWindow(display, win);
                            drawall(-35, 0);
                            printf("Left pressed.\n");
                        }
                        break;
                    case XK_Right:
                        if (up==0) {
                            XClearWindow(display, win);
                            drawall(35, 0);									
                            printf("Right pressed.\n");
                        }
                        break;
                    case XK_Up:
                        XClearWindow(display, win);
                        level++;
                        if (move_frog()!=0) {
                            up = 1;
                            drawall(0, -100);
                        } else {
                            level = -1;
                            frog_x = 245, frog_y = 500;
                        }
                        if (level==4)
                            up = 0;
                        printf("Up pressed. Level = %d\n", level);
                        break;
                    case XK_Down:
                        XClearWindow(display, win);
                        level--;
                        if (move_frog()!=0) {
                            drawall(0, 100);
                        } else {
                            level = -1;
                            frog_x = 245, frog_y = 500;
                            up = 0;
                        }
                        printf("Down pressed. Level = %d\n", level);
                        break;
                }
                if (len>0) {
                    if (buf[0]=='q')
                        halt = 1;
                    if (buf[0]=='r') {
                        XClearWindow(display, win);
                        level = -1;
                        frog_x = 245;
                        frog_y = 500;
                        up = 0;
                    }
                }
                break;
            }
        }
    }
}

int main (int argc, char **argv) {

    int i;
    long tid[5] = {1, 2, 3, 4, 5};

    display = XOpenDisplay(NULL);
    win = XCreateSimpleWindow(display, DefaultRootWindow(display), 0, 0, 600, 600, 0, BLACK, LIGHT_GREEN);
    XMapWindow(display, win);

    river = XCreateGC(display, win, 0, 0);
    tree = XCreateGC(display, win, 0, 0);

    frog = XCreateGC(display, win, 0, 0);
    XSync(display, 0);
    drawall(0, 0);

    /* Initializing threads */
    pthread_t threads[5];
    pthread_attr_t attr;

    pthread_mutex_init(&count_mutex, NULL);
    pthread_cond_init(&count_threshold_cv, NULL);

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &attr, float_wood, (void *) tid[0]);
    pthread_create(&threads[1], &attr, float_wood, (void *) tid[1]);
    pthread_create(&threads[2], &attr, float_wood, (void *) tid[2]);
    pthread_create(&threads[3], &attr, float_wood, (void *) tid[3]);
    pthread_create(&threads[4], &attr, draw_kb, (void *) tid[4]);

    /* wait for threads to complete and clean up, exit */
    for (i=0; i<NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&count_mutex);
    pthread_cond_destroy(&count_threshold_cv);
    pthread_exit(NULL);

    return 0;
}
