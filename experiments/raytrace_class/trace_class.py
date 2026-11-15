"""The ray tracer's trace pass (TRACE, the tree in L1) as a class: the handler and its
words, the tree as a field, and the hand program's moves and walk as inline bodies,
the walk's headers included as they are (build/tracepass/ beside this file).
README.md lists what the class could not say."""

import komorebi as km

TREE = 16  # the tree's pages: the plan's (`made` gives the scene's)
PAGES = 4  # pages of the packet and hit buffers (the hand's kInPages, kOutPages)


class TracePass(km.Program):
    """trace_rays: the hand program's TRACE command on every core, its words in its
    order (a0 directory .. a4 profile), then the tree's tensor; the tree is filled
    into its field by the command itself. fill_tree: that fill alone."""

    TREE = km.symbol()  # the plan's
    tree: km.L1[km.u32, TREE, 32, 32]
    # temporary: the bodies' own buffers, as fields no line fills (see README.md)
    packets_in: km.L1[km.fp32, PAGES, 32, 32]
    hits_out: km.L1[km.fp32, PAGES, 32, 32]
    reader_words: km.L1[km.u32, 1, 32, 32]
    writer_words: km.L1[km.u32, 1, 32, 32]

    @km.handler
    def trace_rays(
        self,
        directory: km.Tensor,
        packets: km.Tensor,
        hits: km.Tensor,
        mode: int,
        profile: km.Tensor,
        image: km.Tensor,
    ):
        self.tree.fill(km.Stream(image.pages(0, TREE)))
        # Mover A: the hand reader's TRACE (its directory entry, a header page to
        # compute, then each packet: 1,408 B inside a DRAM page of 64 packets).
        km.cxx(
            """
            #include "../../../../projects/raytrace/kernels/timed.hpp"
            #include "../../../../projects/raytrace/kernels/dir_noc.hpp"
            // temporary: the tensors' command words and km::Tensor types, the
            // fields' buffers by their names
            const uint32_t started = km::wall();
            const uint32_t directory_space = get_write_ptr(kFieldReaderWords);
            volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
            if (!read_directory<Directory>(
                    movement, command, kHandlerTrace, directory_space
                ) ||
                !timed::reads_done(movement, km::kStMover)) {
                return false;
            }
            for (uint32_t counter = 0; counter < kRdCounters; ++counter) {
                directory[kRdWords + counter] = 0;
            }
            const uint32_t first_packet = directory[kDirFirst];
            const uint32_t packet_count = directory[kDirCount];
            if (!movement.reserve(kFieldPacketsIn, 1)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* header = (
                km::l1(get_write_ptr(kFieldPacketsIn))
            );
            header[kDirFirst] = first_packet;
            header[kDirCount] = packet_count;
            cb_push_back(kFieldPacketsIn, 1);
            uint32_t wait = 0, read = 0;
            const InterleavedAddrGen<true> addresses = {
                .bank_base_address = command.a1,
                .page_size = kPacketBytes * kPagePackets
            };
            for (uint32_t page = 0; page < packet_count; ++page) {
                const uint32_t read_started = km::wall();
                if (!movement.reserve(kFieldPacketsIn, 1)) {
                    return false;
                }
                wait += km::wall() - read_started;
                const uint32_t packet = first_packet + page;
                noc_async_read(
                    get_noc_addr(
                        packet / kPagePackets,
                        addresses,
                        (packet % kPagePackets) * kPacketBytes
                    ),
                    get_write_ptr(kFieldPacketsIn),
                    kPacketBytes
                );
                if (!timed::reads_done(movement, km::kStMover)) {
                    return false;
                }
                cb_push_back(kFieldPacketsIn, 1);
                read += km::wall() - read_started;
            }
            read -= wait;
            directory[kRdWords + 0] = wait;
            directory[kRdWords + 1] = read;
            directory[kRdWords + 3] = packet_count;
            directory[kRdWords + 2] = km::wall() - started;
            """,
            on="mover_a",
        )
        # Compute: unpack hands the tree's and each packet's address to the pack CPU,
        # which walks the packet (walk.hpp) into its hit page, then the profile page.
        km.cxx(
            """
            #include "../../kernels/pack_headers.hpp"
            #include "../../../../projects/raytrace/kernels/raytrace.h"
            #include "../../../../projects/raytrace/kernels/timed.hpp"
            // temporary: the fields' buffers by name (a body's own buffers)
            #ifdef TRISC_UNPACK
            {
                constexpr uint32_t kPack = ckernel::ThreadId::PackThreadId;
                ckernel::mailbox_write(kPack, get_tile_l1_byte_address(tree, 0));
                if (!timed::front(kFieldPacketsIn, 1)) {
                    ckernel::mailbox_write(kPack, timed::kAbort);
                    return;
                }
                const volatile uint32_t* header = (
                    reinterpret_cast<const volatile uint32_t*>(
                        get_tile_l1_byte_address(kFieldPacketsIn, 0)
                    )
                );
                asm volatile("fence" ::: "memory");
                const uint32_t first_packet = header[kDirFirst];
                const uint32_t packets = header[kDirCount];
                ckernel::mailbox_write(kPack, first_packet);
                ckernel::mailbox_write(kPack, packets);
                cb_pop_front(kFieldPacketsIn, 1);
                for (uint32_t page = 0; page < packets; ++page) {
                    if (!timed::front(kFieldPacketsIn, 1)) {
                        ckernel::mailbox_write(kPack, timed::kAbort);
                        return;
                    }
                    ckernel::mailbox_write(
                        kPack, get_tile_l1_byte_address(kFieldPacketsIn, 0)
                    );
                    uint32_t released = 0;
                    const bool handed = timed::mail(kPack, released);
                    cb_pop_front(kFieldPacketsIn, 1);
                    if (!handed) {
                        return;
                    }
                }
            }
            #endif
            #ifdef TRISC_PACK
            {
                using namespace rt;
                constexpr uint32_t kErrorUnpack = 4, kErrorMail = 8, kErrorOut = 16;
                constexpr uint32_t kErrorTree = 128;
                constexpr uint8_t kUnpack = ckernel::ThreadId::UnpackThreadId;
                walk::Profile profile;
                const uint32_t started = wall();
                uint32_t tree_address = timed::kAbort;
                if (!timed::mail(kUnpack, tree_address)) {
                    tree_address = timed::kAbort;
                }
                const uint32_t* image = (
                    tree_address == timed::kAbort
                        ? nullptr
                        : reinterpret_cast<const uint32_t*>(tree_address)
                );
                auto mail = [&](uint32_t& value) {
                    if (!timed::mail(kUnpack, value)) {
                        profile.error |= kErrorMail;
                        return false;
                    }
                    profile.error |= value == timed::kAbort ? kErrorUnpack : 0;
                    return value != timed::kAbort;
                };
                auto out_ptr = [] {
                    return get_local_cb_interface(kFieldHitsOut).fifo_wr_ptr << 4;
                };
                auto push_out = [](uint32_t address, uint32_t words) {
                    asm volatile("fence" ::: "memory");
                    const uint32_t last = (
                        reinterpret_cast<volatile uint32_t*>(address)[words - 1]
                    );
                    asm volatile("and x0, x0, %0" : : "r"(last) : "memory");
                    cb_push_back(kFieldHitsOut, 1);
                };
                auto reserve_out = [&] {
                    const uint32_t reserve_started = wall();
                    const bool reserved = timed::back(kFieldHitsOut, 1);
                    profile.wait_out += wall() - reserve_started;
                    profile.error |= reserved ? 0 : kErrorOut;
                    return reserved ? out_ptr() : 0u;
                };
                auto trace_list = [&](uint32_t packets, bool any) {
                    const uint32_t* nodes = image + image[kTreeNodeOff];
                    const uint32_t* order = image + image[kTreeOrderOff];
                    const uint32_t* tris = image + image[kTreeTriOff];
                    for (uint32_t packet = 0; packet < packets; ++packet) {
                        const uint32_t wait_started = wall();
                        uint32_t address = 0;
                        const bool arrived = mail(address);
                        asm volatile("fence" ::: "memory");
                        profile.wait_in += wall() - wait_started;
                        if (!arrived) {
                            return false;
                        }
                        const uint32_t* page = (
                            reinterpret_cast<const uint32_t*>(address)
                        );
                        uint32_t phase_started = wall();
                        const uint32_t packet_started = phase_started;
                        walk::load_packet(page);
                        const uint32_t signs = page[kVec * kPkHead + kHeadSigns];
                        ckernel::mailbox_write(kUnpack, 1);
                        uint32_t phase_ended = wall();
                        profile.load += phase_ended - phase_started;
                        walk::packet(nodes, order, tris, signs, any, profile);
                        phase_started = wall();
                        tensix_sync();
                        phase_ended = wall();
                        profile.drain += phase_ended - phase_started;
                        const uint32_t destination = reserve_out();
                        if (destination == 0) {
                            return false;
                        }
                        phase_started = wall();
                        push_out(
                            destination,
                            walk::store_hits(
                                reinterpret_cast<uint32_t*>(destination),
                                phase_ended - packet_started,
                                any
                            )
                        );
                        profile.out += wall() - phase_started;
                        profile.packets++;
                    }
                    return true;
                };
                auto write_profile = [&] {
                    if (!timed::back(kFieldHitsOut, 1)) {
                        return;
                    }
                    const uint32_t destination = out_ptr();
                    volatile uint32_t* page = (
                        reinterpret_cast<volatile uint32_t*>(destination)
                    );
                    const uint32_t* counters = (
                        reinterpret_cast<const uint32_t*>(&profile)
                    );
                    for (uint32_t index = 0; index < 32; ++index) {
                        page[index] = index < walk::kProfileWords ? counters[index] : 0;
                    }
                    push_out(destination, 32);
                };
                uint32_t first_packet = 0, packets = 0;
                if (!mail(first_packet) || !mail(packets)) {
                    profile.total = wall() - started;
                    write_profile();
                    return;
                }
                setup();
                if (image == nullptr) {
                    profile.error |= kErrorTree;
                } else {
                    trace_list(packets, mode == kModeAny);
                }
                finish();
                profile.total = wall() - started;
                write_profile();
            }
            #endif
            """,
            takes=[self.tree, mode],
        )
        # Mover B: the hand writer's TRACE (each hit page, an item inside a DRAM page
        # of 64, then the profile page with the reader's and its own counters).
        km.cxx(
            """
            #include "../../../../projects/raytrace/kernels/timed.hpp"
            #include "../../../../projects/raytrace/kernels/dir_noc.hpp"
            // temporary: the tensors' command words and km::Tensor types, the
            // fields' buffers by their names
            const uint32_t started = km::wall();
            const uint32_t directory_space = get_write_ptr(kFieldWriterWords);
            volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
            if (!read_directory<Directory>(
                    movement, command, kHandlerTrace, directory_space
                ) ||
                !timed::reads_done(movement, km::kStMover)) {
                return false;
            }
            const uint32_t first = directory[kDirFirst], count = directory[kDirCount];
            const uint32_t bytes = mode == kModeAny ? kHitAnyBytes : kHitBytes;
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = command.a2, .page_size = bytes * kPagePackets
            };
            uint32_t wait = 0, write = 0;
            for (uint32_t packet = 0; packet < count; ++packet) {
                const uint32_t page_started = km::wall();
                if (!movement.wait_front(kFieldHitsOut, 1)) {
                    return false;
                }
                wait += km::wall() - page_started;
                const uint32_t at = first + packet;
                noc_async_write(
                    get_read_ptr(kFieldHitsOut),
                    get_noc_addr(at / kPagePackets, pages, (at % kPagePackets) * bytes),
                    bytes
                );
                if (!timed::writes_sent(movement, km::kStMover)) {
                    return false;
                }
                cb_pop_front(kFieldHitsOut, 1);
                write += km::wall() - page_started;
            }
            write -= wait;
            if (!movement.wait_front(kFieldHitsOut, 1)) {
                return false;
            }
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* profile = km::l1(get_read_ptr(kFieldHitsOut));
            volatile tt_l1_ptr uint32_t* reader_words = (
                km::l1(get_write_ptr(kFieldReaderWords))
            );
            for (uint32_t counter = 0; counter < kRdCounters; ++counter) {
                profile[kProfReader + counter] = reader_words[kRdWords + counter];
            }
            profile[kProfWriter + 0] = wait;
            profile[kProfWriter + 1] = write;
            profile[kProfWriter + 2] = km::wall() - started;
            profile[kProfWriter + 3] = count;
            if (!km::write_pages(
                movement,
                command,
                Profile(command.a4),
                movement.me(),
                1,
                get_read_ptr(kFieldHitsOut)
            )) {
                return false;
            }
            if (!timed::writes_sent(movement, km::kStMover)) {
                return false;
            }
            cb_pop_front(kFieldHitsOut, 1);
            """,
            takes=[mode],
            on="mover_b",
        )

    @km.handler
    def fill_tree(self, image: km.Tensor):
        self.tree.fill(km.Stream(image.pages(0, TREE)))
        km.cxx("// the tree's fill alone (the account's term)", takes=[self.tree])


# temporary: the compute kernel's options the walk needs, set through the host API (a
# class sets none): the pack CPU's vector unit, DST whole.
WALK_OPTIONS = {"enable_trisc2_rvv": True, "dst_full_sync_en": True}


def made(session, tree_pages, tensors, log):
    """The class made with the scene's tree pages and example tensors of each name,
    one command at a time as the hand program; the compute options the walk needs."""
    program = TracePass(
        log=log, plan={"TREE": tree_pages}, l1=session.l1()[1], depth=1, **tensors
    )
    program.compute.update(WALK_OPTIONS)
    return program
