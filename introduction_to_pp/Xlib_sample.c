#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>

#define WHITE 0xFFFFFF
#define BLACK 0x000000
#define DARK_GREEN 0x009933
#define WOOD_BROWN 0x663300
#define RIVER_BLUE 0x0099FF

int main (int argc, const char **argv) {

    /* Create X connection */
    Display *display = XOpenDisplay(NULL);

    /* Create a window */
    Window win = XCreateSimpleWindow(display, DefaultRootWindow(display), 0, 0, 600, 600, 0, BLACK, DARK_GREEN);

    /* specify connection */
    XMapWindow(display, win);
    /* Flushes the output buffer and then waits until all requests have been received and processed by the X server */
    XSync(display, 0);

    /* Create Graphics Context */
    GC gc = XCreateGC(display, win, 0, 0);
    /* Specify color */
    XSetForeground(display, gc, BLACK);
    /* Draw a rectangle */
    XFillRectangle(display, win, gc, 20, 20, 50, 50);

    /* refresh window, snd the drawing request to X server */
    XFlush(display);


    sleep(100);

    return 0;
}
