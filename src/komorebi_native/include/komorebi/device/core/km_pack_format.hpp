// The packer set for a buffer of a compile-time index (pack CPU): the library's
// pack_reconfig_data_format with the index a template argument, always inline.
#pragma once

#include <stdint.h>

namespace km {

    // Buffer kIndex's formats and tile shape, read as constants as each architecture's
    // llk_pack_reconfig_data_format reads them; nothing off the pack CPU.
    template <uint32_t kIndex>
    ALWI void pack_format() {
#if defined(TRISC_PACK)
        LLK_SAN_FUNCTION();
        constexpr uint32_t src = pack_src_format[kIndex];
        constexpr uint32_t dst = pack_dst_format[kIndex];
        constexpr uint32_t faces = pack_tile_num_faces[kIndex];
#if defined(ARCH_BLACKHOLE)
        constexpr uint32_t columns = pack_tile_c_dim[kIndex];
        SAN_HOOK(reconfigure(
            StateVal<Operand<Exu::Pack>::DestWidth32>(DST_ACCUM_MODE),
            StateVal<Operand<Exu::Pack>::InputFormat>(src),
            StateVal<Operand<Exu::Pack>::OutputFormat>(dst),
            StateVal<Operand<Exu::Pack>::TileWidth>(columns),
            StateVal<Operand<Exu::Pack>::NumFaces>(faces),
            StateVal<Operand<Exu::Pack>::PartialFace>(false),
            StateDiscard<std::uint32_t>(get_local_cb_interface(kIndex).fifo_page_size)
        ));
        _llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
            src,
            dst,
            get_local_cb_interface(kIndex).fifo_page_size,
            columns,
            faces,
            false
        );
#elif defined(ARCH_WORMHOLE)
        constexpr uint32_t rows = pack_tile_face_r_dim[kIndex];
        constexpr bool partial = pack_partial_face[kIndex];
        constexpr bool narrow = pack_narrow_tile[kIndex];
        SAN_HOOK(reconfigure(
            StateVal<Operand<Exu::Pack>::DestWidth32>(DST_ACCUM_MODE),
            StateVal<Operand<Exu::Pack>::InputFormat>(src),
            StateVal<Operand<Exu::Pack>::OutputFormat>(dst),
            StateVal<Operand<Exu::Pack>::FaceHeight>(rows),
            StateVal<Operand<Exu::Pack>::NumFaces>(faces),
            StateVal<Operand<Exu::Pack>::PartialFace>(partial),
            StateVal<Operand<Exu::Pack>::NarrowTile>(narrow),
            StateDiscard<std::uint32_t>(get_local_cb_interface(kIndex).fifo_page_size)
        ));
        _llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
            src,
            dst,
            get_local_cb_interface(kIndex).fifo_page_size,
            rows,
            faces,
            partial,
            narrow
        );
#else
#error "km::pack_format: Blackhole or Wormhole"
#endif
#endif
    }

}  // namespace km
