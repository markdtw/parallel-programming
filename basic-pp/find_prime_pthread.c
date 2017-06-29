#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS 3

int is_prime[1000000];
int TCOUNT=0, COUNT_LIMIT=0;
int count = 0, latest_prime = 2;

pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

void set_up_primetable () {
    int i, j;
    for (i=0; i<1000000; i++)
        is_prime[i] = 1;

    is_prime[0] = is_prime[1] = 0;
    for (i=2; i<1000; i++) {
        if (is_prime[i]==1) {
            for (j=2; j<=1000000/i; j++)
                is_prime[i*j] = 0;
        }
    }
}
void *prime_count (void *t) {
    int i, j;
    double result = 0.0;
    long my_id = (long) t;

    while (count < TCOUNT) {

        pthread_mutex_lock(&count_mutex);
        if (is_prime[count]==1)
            latest_prime = count;

        count++;

        /* 
         * Check the value of count and signal waiting thread when condition is reached.
         * Note that this occurs while mutex is locked.
         */

        if (count==COUNT_LIMIT) {
            printf("prime_count(): thread %ld, p = %d, prime reached.\n", my_id, count);
            pthread_cond_signal(&count_threshold_cv);
            printf("Just sent signal.\n");
        } else {
            printf("prime_count(): thread %ld, p = %d.\n", my_id, count);
            if (is_prime[count]==1)
                printf("prime_count(): thread %ld, find prime = %d.\n", my_id, count);

        }
        pthread_mutex_unlock(&count_mutex);

        sleep(1);
        if (count==COUNT_LIMIT)
            sleep(1);
    }
    pthread_exit(NULL);
}
void *watch_count (void *t) {
    long my_id = (long) t;
    printf("Starting watch_count(): thread %ld\n", my_id);

    /* Lock mutex and wait for signal.
     * Note that the pthread_cond_wait routine will automatically and
     * atomically unlock mutex while it waits.
     * Also, note that if COUNT_LIMIT is reached before this
     *  routine is run by the waiting thread, the loop will be skipped to prevent 
     *  pthread_cond_wait from never returning.
     */
    pthread_mutex_lock(&count_mutex);
    while  (count<COUNT_LIMIT) {
        printf("watch_count(): thread %ld, p = %d. Going into wait...\n", my_id, count);
        pthread_cond_wait(&count_threshold_cv, &count_mutex);
        printf("watch_count(): thread %ld Condition signal received. p = %d.\n", my_id, count);
        count += latest_prime;
        printf("watch_count(): thread %ld Updating the value of p...\n", my_id);
        printf("the latest prime found before p = %d.\n", latest_prime);
        printf("watch_count(): thread %ld Count p now = %d.\n", my_id, count);
    }

    printf("watch_count(): thread %ld Unlocking mutex.\n", my_id);
    pthread_mutex_unlock(&count_mutex);
    pthread_exit(NULL);
}
int main (int argc, char **argv) {
    if (argc!=3) {
        printf("usage: ./a.out <tcount> <count_limit>\n");
        exit(-1);
    }
    int i, rc;
    long t1 = 1, t2 = 2, t3 = 3;
    pthread_t threads[3];
    pthread_attr_t attr;

    TCOUNT = atoi(argv[1]);
    COUNT_LIMIT = atoi(argv[2]);

    set_up_primetable();
    /* 1. Initialize mutex and condition variable objects. */
    pthread_mutex_init(&count_mutex, NULL);
    pthread_cond_init(&count_threshold_cv, NULL);

    /* For portability, explicitly create threads in a joinable state. */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&threads[0], &attr, watch_count, (void *) t1);
    pthread_create(&threads[1], &attr, prime_count, (void *) t2);
    pthread_create(&threads[2], &attr, prime_count, (void *) t3);

    /* 2. Wait for all threads to complete. */
    for (i=0; i<NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    printf("Main(): Waited and joined with %d threads. Final value of count = %d. Done.\n", NUM_THREADS, count);

    /* 3. Clean up and exit. */
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&count_mutex);
    pthread_cond_destroy(&count_threshold_cv);
    pthread_exit(NULL);

    return 0;
}
