// The simulator: the grid's set-up and run, fault injection and the state readers.
// One thread runs at a time, in a fixed rotation.
#include <stdint.h>

#include <cstring>

#include "sim_impl.hpp"

namespace kmsim {

    volatile uint32_t wall_clock[2];

    Sim::Sim(Config c) : cfg(std::move(c)), impl_(std::make_unique<Impl>(*this, cfg)) {
        impl_->setup();
    }

    Sim::~Sim() {
        munmap(impl_->region, static_cast<size_t>(impl_->n + 1) * kL1Bytes);
        g = nullptr;
    }

    void Sim::run() {
        g = impl_.get();
        wall_clock[0] = 0;
        wall_clock[1] = 0;
        impl_->run();
    }

    void Sim::freeze(uint32_t core, Role role, uint64_t cycles) {
        for (auto& t : impl_->threads) {
            if (t->kernel && t->core == core && t->role == role) {
                t->frozen_until = impl_->clock + cycles;
            }
        }
    }

    void Sim::set_stop(uint32_t core) {
        *reinterpret_cast<uint32_t*>(
            impl_->cores[core].mem + kCtlOff + KM_CTL_STOP
        ) = 1;
        impl_->log(core, kEvStop, 0, ~0u);
    }

    uint64_t Sim::now() const {
        return impl_->clock;
    }

    uint32_t Sim::word(uint32_t core, uint32_t off) const {
        uint32_t v;
        std::memcpy(&v, impl_->cores[core].mem + kCtlOff + off, 4);
        return v;
    }

    uint32_t Sim::mailbox_words() const {
        uint32_t n = 0;
        for (const auto& c : impl_->cores) {
            for (const auto& row : c.mbox) {
                for (const auto& q : row) {
                    n += static_cast<uint32_t>(q.size());
                }
            }
        }
        return n;
    }

    uint32_t Sim::popped(uint32_t core, uint32_t cb) const {
        return impl_->cores[core].cb[cb & 31].acked;
    }

    uint32_t Sim::state(uint32_t core, uint32_t index) const {
        return word(core, KM_CTL_STATE + 4 * index);
    }

}  // namespace kmsim
