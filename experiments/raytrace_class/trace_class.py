"""The ray tracer's trace pass (TRACE, the tree in L1) as a class: the tree loaded once
and kept, the directory's words, the packets and hit pages as items of the tensors'
pages, and the hand program's walk as inline bodies on the pack CPU, its headers
included as they are. README.md says what each line lowers to."""

import komorebi as km

TREE = 16  # the tree's pages: the plan's (`made` gives the scene's)
PACKET = 1408  # a packet (kPacketBytes), 64 to a page of the packets tensor
HIT = 2432  # a nearest-hit page (kHitBytes), 64 to a page of the hits tensor
# The walk's compute options: the pack CPU's vector unit, DST whole.
WALK = {"enable_trisc2_rvv": True, "dst_full_sync_en": True}


class TracePass(km.Program, compute=WALK):
    """load_tree: the tree into its field, kept. trace_rays: the hand program's
    TRACE command on every core, its words in its order (a0 directory .. a4
    profile): each core's packets walked into their hit pages, then its profile."""

    TREE = km.symbol()  # the plan's
    tree: km.L1[km.u32, TREE, 32, 32]

    @km.handler
    def load_tree(self, image: km.Tensor):
        self.tree.fill(km.Stream(image.pages(0, TREE)))

    @km.handler
    def trace_rays(
        self,
        directory: km.Tensor,
        packets: km.Tensor,
        hits: km.Tensor,
        mode: int,
        profile: km.Tensor,
    ):
        first, count = directory.pages(self.rank, 1).read()[0:2]
        hit = km.Stream(hits.items(first, count, size=HIT), write=True, pattern="B")
        image = km.cxx("image = tree;", takes=[self.tree], gives=int, on="pack")
        # The walk's state on the pack CPU, seen by its bodies below.
        km.cxx(
            """
            #include "../../../../projects/raytrace/kernels/raytrace.h"
            using namespace rt;
            constexpr uint32_t kErrorOut = 16;
            walk::Profile walk_profile;
            const uint32_t started = wall();
            const uint32_t* tree_words = reinterpret_cast<const uint32_t*>(image);
            const uint32_t* nodes = tree_words + tree_words[kTreeNodeOff];
            const uint32_t* order = tree_words + tree_words[kTreeOrderOff];
            const uint32_t* tris = tree_words + tree_words[kTreeTriOff];
            setup();
            """,
            takes=[image],
            on="pack",
            include=["../../../../projects/raytrace/kernels/walk.hpp"],
        )
        for rays in km.Stream(packets.items(first, count, size=PACKET)):
            km.cxx(
                """
                const uint32_t* page = reinterpret_cast<const uint32_t*>(rays);
                const bool any = mode == kModeAny;
                uint32_t phase_started = wall();
                const uint32_t packet_started = phase_started;
                walk::load_packet(page);
                const uint32_t signs = page[kVec * kPkHead + kHeadSigns];
                uint32_t phase_ended = wall();
                walk_profile.load += phase_ended - phase_started;
                walk::packet(nodes, order, tris, signs, any, walk_profile);
                phase_started = wall();
                tensix_sync();
                phase_ended = wall();
                walk_profile.drain += phase_ended - phase_started;
                phase_started = wall();
                if (hit == 0) {
                    walk_profile.error |= kErrorOut;
                } else {
                    uint32_t* out = reinterpret_cast<uint32_t*>(hit);
                    const uint32_t words = walk::store_hits(
                        out, phase_ended - packet_started, any
                    );
                    asm volatile("fence" ::: "memory");
                    const uint32_t last = (
                        reinterpret_cast<volatile uint32_t*>(out)[words - 1]
                    );
                    asm volatile("and x0, x0, %0" : : "r"(last) : "memory");
                }
                walk_profile.out += wall() - phase_started;
                walk_profile.packets++;
                """,
                takes=[rays, hit, mode],
                on="pack",
            )
        report = km.Stream(profile.pages(self.rank, 1), write=True, pattern="B")
        km.cxx(
            """
            finish();
            walk_profile.total = wall() - started;
            if (report != 0) {
                volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(report);
                const uint32_t* counters = (
                    reinterpret_cast<const uint32_t*>(&walk_profile)
                );
                for (uint32_t index = 0; index < kOutPageBytes / 4; ++index) {
                    words[index] = index < walk::kProfileWords ? counters[index] : 0;
                }
                asm volatile("fence" ::: "memory");
                const uint32_t last = words[kOutPageBytes / 4 - 1];
                asm volatile("and x0, x0, %0" : : "r"(last) : "memory");
            }
            """,
            takes=[report],
            on="pack",
        )


def made(session, tree_pages, tensors, log):
    """The class made with the scene's tree pages and example tensors of each name,
    one command at a time as the hand program."""
    return TracePass(
        log=log, plan={"TREE": tree_pages}, l1=session.l1()[1], depth=1, **tensors
    )
