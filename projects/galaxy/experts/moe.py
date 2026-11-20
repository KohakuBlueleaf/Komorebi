"""The router and the experts of a layer's feed-forward as classes that read the weights
where the loader put them: a weight is a run of one-tile pages of its island's arena,
row-major tiles of its held [in, out] shape, and a row of tiles is a strided walk of the
arena. README.md."""

import komorebi as km

D_TILES = 160  # the model's hidden, 5,120
HID = 72  # an expert's width, 2,304
SCORES = 12  # the router's 384 scores


class Pick(km.Entry):
    w1: int
    w3: int
    w2: int


class Moe(
    km.Program,
    compute={
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
):
    """The router (route: the logits), the routed experts (bfp4_b, `w`) and the shared
    expert (bfp8_b, `w8`; the word `eight`) of a layer as one program, kept loaded. The
    experts' list has `e` places (a word): up (h | g), gate, down. README.md."""

    K_TILES = km.constant(160)
    HID = km.constant(72)
    OUT = km.constant(160)
    SCORES = km.constant(12)
    SPLIT = km.constant(5)  # the router's K split, divides 160
    SPLIT_U = km.constant(2)
    WU = km.constant(4)
    GQ = km.constant(4)
    PG = km.constant(13)  # the gate's place groups: GQ x PG cores at most
    SPLIT_D = km.constant(2)
    WD = km.constant(5)
    CHUNK = km.constant(2)
    U_BLOCK = CHUNK * WU
    D_BLOCK = CHUNK * WD
    KR = K_TILES // SPLIT
    KS = K_TILES // SPLIT_U  # SPLIT_U divides 160
    RS = HID // SPLIT_D  # SPLIT_D divides 72, RS even
    QS = HID // GQ  # GQ divides 72
    # the sums of up and route in float32 fields (the form verified on the cards)
    up_sums: km.L1[km.fp32, 1, WU, 32, 32]
    route_sums: km.L1[km.fp32, 1, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.tile_function
    def swiglu(h, g, r):
        return km.silu(km.minimum(h, 10.0)) * km.minimum(km.maximum(g, -10.0), 10.0) * r

    @km.handler
    def route(self, x: km.Tensor, wg: km.Tensor, logits: km.Tensor, first: int):
        share = self.rank % self.SPLIT
        c = self.rank // self.SPLIT
        k0 = share * self.KR
        start = first + c + k0 * self.SCORES
        for xb, wb in zip(
            km.Stream(x.tiles[0:1, k0 : k0 + self.KR], block=self.CHUNK),
            km.Stream(
                wg.tiles[start : start + self.KR * self.SCORES : self.SCORES, 0:1],
                block=self.CHUNK,
            ),
        ):
            self.route_sums += self.project(xb, wb)
        whole = km.reduce(self.route_sums, form="tree", group=self.SPLIT, local=True)
        if self.rank == self.leader:
            logits.tiles[0:1, c : c + 1].write(whole)

    @km.handler
    def up(
        self,
        x: km.Tensor,
        w: km.Tensor,
        w8: km.Tensor,
        table: km.Tensor,
        hg: km.Tensor,
        e: int,
        eight: int,
    ):
        share = self.rank % self.SPLIT_U
        c = self.rank // self.SPLIT_U * self.WU  # of an expert's [h | g], 144 columns
        half = c // self.HID  # 0: h (w1), 1: g (w3)
        cc = c - half * self.HID
        k0 = share * self.KS
        for pos in range(e):
            for pick in table.pages(pos, 1).entries(Pick, size=3, fixed=1):
                start = pick.w1 + half * (pick.w3 - pick.w1) + cc + k0 * self.HID
                if eight == 0:
                    for xb, wb in zip(
                        km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=self.CHUNK),
                        km.Stream(
                            w.tiles[
                                start : start + self.KS * self.HID : self.HID,
                                0 : self.WU,
                            ],
                            block=self.U_BLOCK,
                        ),
                    ):
                        self.up_sums += self.project(xb, wb)
                else:
                    for xb, wb in zip(
                        km.Stream(x.tiles[0:1, k0 : k0 + self.KS], block=self.CHUNK),
                        km.Stream(
                            w8.tiles[
                                start : start + self.KS * self.HID : self.HID,
                                0 : self.WU,
                            ],
                            block=self.U_BLOCK,
                        ),
                    ):
                        self.up_sums += self.project(xb, wb)
                whole = km.reduce(
                    self.up_sums, form="tree", group=self.SPLIT_U, local=True
                )
                if self.rank == self.leader:
                    at = pos * 2 * self.HID + c
                    hg.tiles[0:1, at : at + self.WU].write(whole)

    @km.handler
    def gate(self, hg: km.Tensor, f: km.Tensor, hidden: km.Tensor, e: int):
        group = self.rank // self.GQ  # places group, group + PG, ... of the list
        a = (self.rank - group * self.GQ) * self.QS
        for turn in range((e - group + self.PG - 1) // self.PG):
            pos = group + turn * self.PG
            h = pos * 2 * self.HID + a
            g = h + self.HID
            o = pos * self.HID + a
            out = km.Stream(hidden.tiles[0:1, o : o + self.QS], write=True, block=1)
            for hb, gb, rb in zip(
                km.Stream(hg.tiles[0:1, h : h + self.QS], block=1),
                km.Stream(hg.tiles[0:1, g : g + self.QS], block=1),
                km.Stream(f.tiles[0:1, o : o + self.QS], block=1),
            ):
                out.push(self.swiglu(hb, gb, rb))

    @km.handler
    def down(
        self,
        hidden: km.Tensor,
        w: km.Tensor,
        w8: km.Tensor,
        table: km.Tensor,
        y: km.Tensor,
        e: int,
        eight: int,
    ):
        share = self.rank % self.SPLIT_D
        first = self.rank // self.SPLIT_D * self.WD
        r0 = share * self.RS
        acc = km.tile.zeros(km.fp32, self.WD)
        for pos in range(e):
            for pick in table.pages(pos, 1).entries(Pick, size=3, fixed=1):
                start = pick.w2 + first + r0 * self.OUT
                h0 = pos * self.HID + r0
                if eight == 0:
                    for xb, wb in zip(
                        km.Stream(
                            hidden.tiles[0:1, h0 : h0 + self.RS], block=self.CHUNK
                        ),
                        km.Stream(
                            w.tiles[
                                start : start + self.RS * self.OUT : self.OUT,
                                0 : self.WD,
                            ],
                            block=self.D_BLOCK,
                        ),
                    ):
                        acc += self.project(xb, wb)
                else:
                    for xb, wb in zip(
                        km.Stream(
                            hidden.tiles[0:1, h0 : h0 + self.RS], block=self.CHUNK
                        ),
                        km.Stream(
                            w8.tiles[
                                start : start + self.RS * self.OUT : self.OUT,
                                0 : self.WD,
                            ],
                            block=self.D_BLOCK,
                        ),
                    ):
                        acc += self.project(xb, wb)
        whole = km.reduce(acc, form="tree", group=self.SPLIT_D, local=True)
        if self.rank == self.leader:
            y.tiles[0:1, first : first + self.WD].write(whole)


def cores(fields, e):
    """The cores of each handler of Moe for its constants (`fields`) and a list of e."""
    return {
        "route": 12 * fields["SPLIT"],
        "up": 2 * HID // fields["WU"] * fields["SPLIT_U"],
        "gate": fields["GQ"] * min(e, fields["PG"]),
        "down": D_TILES // fields["WD"] * fields["SPLIT_D"],
    }
