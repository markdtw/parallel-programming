#include <X11/Xlib.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

typedef struct complexType{
    double real, imag;
}Compl;

pthread_mutex_t mutex, ptr_mutex;
Display *display;
GC gc;
Window window;
int width = 400;
int height = 400;
int ptr = 0;
void *draw_mandelbrot (void *t) {
    while (ptr<400) {
        long id = (long) t;
        int i = ptr;
        int j;

        /* draw points */
        Compl z, c;
        int repeats;
        double temp, lengthsq;

        for (j=0; j<height; j++) {
            z.real = 0.0;
            z.imag = 0.0;
            c.real = -2.0 + (double)i * (4.0/(double)width);
            c.imag = -2.0 + (double)j * (4.0/(double)height);
            repeats = 0;
            lengthsq = 0.0;

            while (repeats < 100000 && lengthsq < 4.0) {
                // Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4
                temp = z.real*z.real - z.imag*z.imag + c.real;
                z.imag = 2*z.real*z.imag + c.imag;
                z.real = temp;
                lengthsq = z.real*z.real + z.imag*z.imag;
                repeats++;
            }
            pthread_mutex_lock(&mutex);	
            XSetForeground(display, gc, 1024*1024*(repeats%256));
            XDrawPoint(display, window, gc, i, j);
            pthread_mutex_unlock(&mutex);
        }
        pthread_mutex_lock(&ptr_mutex);
        ptr++;
        pthread_mutex_unlock(&ptr_mutex);
    }

    pthread_exit(NULL);
}
int main (void) {

    int screen;			// which screen

    /* open connection with the server */
    display = XOpenDisplay(NULL);
    if(display == NULL) {
        fprintf(stderr, "cannot open display\n");
        return 0;
    }

    screen = DefaultScreen(display);

    /* set window position */
    int x = 0;
    int y = 0;

    /* border width in pixels */
    int border_width = 0;

    /* create window */
    window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y,
        width, height, border_width,
            BlackPixel(display, screen), WhitePixel(display, screen));

    /* create graph */
    XGCValues values;
    long valuemask = 0;

    gc = XCreateGC(display, window, valuemask, &values);
    //XSetBackground (display, gc, WhitePixel (display, screen));
    XSetForeground (display, gc, BlackPixel (display, screen));
    XSetBackground(display, gc, 0X0000FF00);
    XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);

    /* map(show) the window */
    XMapWindow(display, window);
    XSync(display, 0);

    pthread_t threads[width];
    pthread_attr_t attr;

    /* 1. Initialize mutex variable objects. */
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&ptr_mutex, NULL);

    /* For portability, explicitly create threads in a joinable state. */
    struct timespec t1, t2;
    clock_gettime(CLOCK_REALTIME, &t1);

    int i, j;
    long t;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for (t=0; t<40; t++)
        pthread_create(&threads[t], &attr, draw_mandelbrot, (void *) t);

    /* Wait for all threads to complete. */
    for (i=0; i<40; i++)
        pthread_join(threads[i], NULL);

    /* Clean up and exit. */
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&ptr_mutex);

    /* end of record */
    clock_gettime(CLOCK_REALTIME, &t2);
    double timedif = 1000000*(t2.tv_sec-t1.tv_sec)+(t2.tv_nsec-t1.tv_nsec)/1000;
    printf("It took %.5lf seconds to finish dynamic pthread calculation.\n", timedif/1000000);
    printf("Going into sleep...\n");

    XFlush(display);
    sleep(3);
    return 0;
}
