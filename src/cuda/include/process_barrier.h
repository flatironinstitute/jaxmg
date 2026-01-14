#ifndef JAXMG_PROCESS_BARRIER_H
#define JAXMG_PROCESS_BARRIER_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <pthread.h>

class DynamicBarrier
{
    struct Shared
    {
        pthread_mutex_t mtx;
        pthread_cond_t cv;
        int n;      // total participants
        int count;  // remaining to arrive in this generation
        int gen;    // generation number
        int inited; // 0 until fully initialized (published last)
        int rounds;
    };

    std::string shm_name;
    Shared *s = nullptr;
    bool i_created = false;
    int world_size = 0;

    static void throw_errno(const char *what)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
        throw std::runtime_error(buf);
    }
    void snapshot(const char *tag)
    {
        std::printf("[Barrier] %s pid=%d name=%s n=%d count=%d gen=%d rounds=%d inited=%d\n",
                    tag,
                    static_cast<int>(getpid()),
                    shm_name.c_str(),
                    s->n, s->count, s->gen, s->rounds, s->inited);
    }

public:
    DynamicBarrier(int world_size, const std::string &name = "/mpbarrier")
        : shm_name(name), world_size(world_size)
    {
        int fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0644);
        if (fd >= 0)
        {
            i_created = true;
            if (ftruncate(fd, sizeof(Shared)) == -1)
            {
                close(fd);
                throw_errno("ftruncate");
            }
        }
        else
        {
            if (errno != EEXIST)
                throw_errno("shm_open(O_CREAT|O_EXCL)");
            fd = shm_open(shm_name.c_str(), O_RDWR, 0644);
            if (fd == -1)
                throw_errno("shm_open");
        }

        void *addr = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED)
            throw_errno("mmap");
        s = static_cast<Shared *>(addr);
        // snapshot("init");

        if (i_created)
        {
            pthread_mutexattr_t ma;
            pthread_condattr_t ca;
            if (pthread_mutexattr_init(&ma))
                throw std::runtime_error("pthread_mutexattr_init");
            if (pthread_mutexattr_setpshared(&ma, PTHREAD_PROCESS_SHARED))
                throw std::runtime_error("pthread_mutexattr_setpshared");
            if (pthread_condattr_init(&ca))
                throw std::runtime_error("pthread_condattr_init");
            if (pthread_condattr_setpshared(&ca, PTHREAD_PROCESS_SHARED))
                throw std::runtime_error("pthread_condattr_setpshared");

            if (pthread_mutex_init(&s->mtx, &ma))
                throw std::runtime_error("pthread_mutex_init");
            if (pthread_cond_init(&s->cv, &ca))
                throw std::runtime_error("pthread_cond_init");
            pthread_mutexattr_destroy(&ma);
            pthread_condattr_destroy(&ca);

            s->n = world_size;
            s->count = world_size;
            s->gen = 0;
            s->rounds = 0;

            __sync_synchronize();
            s->inited = 1;
            // snapshot("created");
        }
        else
        {
            while (__atomic_load_n(&s->inited, __ATOMIC_ACQUIRE) == 0)
            {
                sched_yield();
            }
            // Fail fast on stale segments: a fresh barrier must start at gen=0
            if (s->gen != 0 || s->rounds != 0)
            {
                std::fprintf(stderr,
                             "[Barrier] stale on attach pid=%d name=%s n=%d count=%d gen=%d rounds=%d inited=%d (expected gen=0)\n",
                             static_cast<int>(getpid()), shm_name.c_str(), s->n, s->count, s->gen, s->rounds, s->inited);
                throw std::runtime_error("Barrier stale segment: non-zero generation on attach");
            }
            if (s->n != world_size)
                throw std::runtime_error("Barrier world_size mismatch across processes.");
        }
    }

    void arrive_and_wait()
    {
        pthread_mutex_lock(&s->mtx);
        // snapshot("arrive");
        int my_gen = s->gen;
        if (--s->count == 0)
        {
            s->gen++;
            s->count = s->n;
            s->rounds++;
            pthread_cond_broadcast(&s->cv);
            pthread_mutex_unlock(&s->mtx);
            // snapshot("release");
        }
        else
        {
            while (my_gen == s->gen)
            {
                pthread_cond_wait(&s->cv, &s->mtx);
            }
            pthread_mutex_unlock(&s->mtx);
            // snapshot("resume");
        }
    }

    void close_and_wait()
    {
        // Make sure all processes are done using the barrier
        arrive_and_wait();

        // Unlink exactly once
        if (i_created)
        {
            shm_unlink(shm_name.c_str());
        }

        // Ensure nobody can race ahead and recreate/open before unlink finishes
        arrive_and_wait();

        // Now it’s safe for everyone to unmap
        if (s)
        {
            munmap(s, sizeof(Shared));
            s = nullptr;
        }
    };
    ~DynamicBarrier()
    {
        if (s)
        {
            munmap(s, sizeof(Shared));
            s = nullptr;
        }
    }
};

#endif // JAXMG_PROCESS_BARRIER_H
