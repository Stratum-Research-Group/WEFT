#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Standalone C++ tool to inspect /dev/shm/weft_spec_ring status

struct RingHeader {
    uint64_t head;
    uint64_t tail;
};

int main(int argc, char** argv) {
    const char* shm_path = "/dev/shm/weft_spec_ring";
    if (argc > 1) shm_path = argv[1];

    std::cout << "==========================================================" << std::endl;
    std::cout << "  WEFT Zero-Copy SHM Ring Status Monitor" << std::endl;
    std::cout << "==========================================================" << std::endl;

    int fd = open(shm_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "[ERROR] Unable to open shared memory file: " << shm_path << std::endl;
        return 1;
    }

    struct stat st;
    fstat(fd, &st);

    void* addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        std::cerr << "[ERROR] mmap failed on: " << shm_path << std::endl;
        close(fd);
        return 1;
    }

    const RingHeader* header = static_cast<const RingHeader*>(addr);
    uint64_t head = header->head;
    uint64_t tail = header->tail;
    size_t capacity = 1024;
    size_t occupancy = (tail > head) ? static_cast<size_t>(tail - head) : 0;
    double pct = (occupancy * 100.0) / capacity;

    std::cout << "[SHM PATH]     : " << shm_path << std::endl;
    std::cout << "[FILE SIZE]    : " << st.st_size << " bytes" << std::endl;
    std::cout << "[HEAD POINTER] : " << head << std::endl;
    std::cout << "[TAIL POINTER] : " << tail << std::endl;
    std::cout << "[OCCUPANCY]    : " << occupancy << " / " << capacity << " (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
    std::cout << "[STATUS]       : " << (occupancy > 0 ? "ACTIVE (Unconsumed nodes present)" : "CLEAN (Empty / Synchronized)") << std::endl;

    munmap(addr, st.st_size);
    close(fd);
    return 0;
}
