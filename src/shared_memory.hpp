#pragma once
#include "types.h"
#include "sys/shm.h"



namespace consumer
{

struct Buffer
{
    std::shared_mutex mutex;
    size_t size;
    data_ptr_t data;
}

class SharedMemory
{
public:
    SharedMemory(int key, size_t size)
    {
        int shmid = shmget(KEY, SIZE, 0666);
        char* shmBuff_ = static_cast<char*>(shmat(shmid, nullptr, 0));
        if (shmBuff_ == reinterpret_cast<Buffer*>(-1))
            std::cerr << "Failed to attach to shared memory.\n";
    }

    ~SharedMemory()
    {
        shmdt(shmBuff_);
    }

    data_ptr_t* GetBuffer() const
    {
        return shmBuff_;
    }


private:
    Buffer* shmBuff_;


};

}



