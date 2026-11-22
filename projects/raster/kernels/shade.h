// SHADE, between FILL and SHOW on a shaded frame: each core colours its tiles in
// place from materials and textures (reference/raster.py rasterize_shaded).
#pragma once

#include <stdint.h>

// SHADE: a0 directory, a1 lists (FILL's), a2 texture pages, a3 table, a4 profile (0:
// none), a6 the opaque layer's directory (0: none; else a0, a1 the masked layer's).
constexpr uint32_t kHandlerShade = 5;  // lit: a7 the surface directory, a5 camera, a8 bank
// LIGHT: a0 directory (the opaque layer's), a7 the surface directory (which names the
// lights' table and the profile tensor): each item's lit colour from its surface.
constexpr uint32_t kHandlerLight = 6;
// With a6, a pixel nearer than the opaque layer's (a tie to it) with alpha byte >=
// kAlphaOpaque takes the masked colour, else the opaque one, in the opaque frame.

// Buffers: the reader's planes, the table, the pack thread's page requests, the
// reader's answers, the page slots (plain memory), the pack thread's own space.
constexpr uint32_t kCbShadePlanes = 14;
constexpr uint32_t kShadePlanePages = 20;  // items of five planes, of ten (a6), LIGHT's
constexpr uint32_t kCbShadeTable = 15;
constexpr uint32_t kShadeTableBytes = 16384;
constexpr uint32_t kCbShadeAsk = 20;
constexpr uint32_t kShadeAskBytes = 1024;
constexpr uint32_t kShadeAskPages = 2;
constexpr uint32_t kCbShadeDone = 21;
constexpr uint32_t kShadeDoneBytes = 64;
constexpr uint32_t kShadeDonePages = 2;
constexpr uint32_t kCbShadeSlots = 22;
constexpr uint32_t kTexPageBytes = 4096;  // 32 x 32 texels, RGBA8, rows from the top
constexpr uint32_t kShadeSlots = 128;     // and one more page: the white texel
constexpr uint32_t kCbShadeWork = 23;
constexpr uint32_t kShadeWorkBytes = 8192;
constexpr uint32_t kShadeColourPages = 3;  // red, green, blue out (kCbOut)
// An item's pages in kCbOut: FILL's five (the first three used), so that the ring
// stays where FILL's items expect it and no item wraps (raster.h kOutPages).
constexpr uint32_t kShadeOutPages = 5;

// Table words: counts, texel tables, background, camera; a material: kind | (1 + the
// albedo's level 0 entry, 0: none) << 8, base rgb, ...; an entry: page, w | h << 16.
constexpr uint32_t kTabMaterials = 0;
constexpr uint32_t kTabEntries = 1;
constexpr uint32_t kTabPages = 2;
constexpr uint32_t kTabGamma = 16;        // TEXEL_LINEAR
constexpr uint32_t kTabSigned = 272;      // TEXEL_SIGNED
constexpr uint32_t kTabUnit = 528;        // TEXEL_UNIT
constexpr uint32_t kTabBackground = 784;  // FILL's background rgb
constexpr uint32_t kTabCamera = 800;      // lit: the reader's copy of the camera
constexpr uint32_t kCamWordsShade = 16;   // reference/scenes.py
constexpr uint32_t kTabMaterial = 816;
constexpr uint32_t kMaterialWords = 14;
constexpr uint32_t kMaterialMask = 4;      // 1 + mask's entry (sizes the albedo's)
constexpr uint32_t kMaterialNormal = 5;    // 1 + normal's entry
constexpr uint32_t kMaterialSpecular = 6;  // 1 + specular's entry
constexpr uint32_t kMaterialLitBase = 7;   // rgb, emissive: x emissive_scale
constexpr uint32_t kMaterialSpecularColour = 10;  // rgb, 0: no specular term
constexpr uint32_t kEntryWords = 2;
// kinds (reference/raster.py): lit is shaded as textured in this step; masked as
// textured, its holes by the masked layer's SHADE (a6)
constexpr uint32_t kKindFlat = 0;
constexpr uint32_t kKindTextured = 1;
constexpr uint32_t kKindMasked = 2;
constexpr uint32_t kKindLit = 3;
constexpr uint32_t kKindEmissive = 4;
constexpr uint32_t kAlphaOpaque = 128;  // a masked texel under it is a hole

// Record word (SETUP writes it on a shaded frame): material | albedo level << 16 |
// normal level << 20 | specular level << 24.
constexpr uint32_t kRecShade = 30;
constexpr uint32_t kShadeLevelShift = 16;
constexpr uint32_t kShadeNormalLevelShift = 20;
constexpr uint32_t kShadeSpecularLevelShift = 24;

// Lit SHADE's frame planes: 1/z (merged), the material word material | kind << 16 in
// the triangle's plane (empty: FILL's -1, bit 31 set); planes 1 to 3 are LIGHT's.
constexpr uint32_t kMaterialNone = 0x80000000;
constexpr uint32_t kMaterialKindShift = 16;
// Its surface planes, two frames like the fill's: a (albedo x base, linear) rgb, normal
// x, y | normal z, visibility, specular rgb (x its texel; 0: no specular term).
constexpr uint32_t kSurfacePlanes = 10;
// The surface's directory page: the first frame's words at kDirFrame, the second's
// here.
constexpr uint32_t kSurfaceSecond = 0;
constexpr uint32_t kSurfaceWords = 32;  // both frames' words (kDirFrameMax + 1 each)
constexpr uint32_t kSurfaceLightTable = 16;    // LIGHT's table tensor
constexpr uint32_t kSurfaceLightProfile = 17;  // LIGHT's profile tensor (0: none)
constexpr uint32_t kSurfaceAlbedo = 0;
constexpr uint32_t kSurfaceNormal = 3;
constexpr uint32_t kSurfaceVisibility = 6;
constexpr uint32_t kSurfaceSpecular = 7;
// A lit item goes out in three parts of kShadeOutPages: the frame's planes (0 and 4
// used), the surface's planes 0 to 4, 5 to 9.
constexpr uint32_t kLitParts = 3;
// The pack thread's lit space (plain memory), in words: per pixel (32 a row) texel
// addresses and slot offsets; a row's keys; the winners' table; the slots' words.
constexpr uint32_t kCbLightWork = 24;
constexpr uint32_t kLightWorkBytes = 102400;
constexpr uint32_t kLwIndex = 0;        // albedo, normal, specular addresses: 3 x 1,024
constexpr uint32_t kLwSlot = 3072;      // the pixel's slot's byte offset (64 a slot)
constexpr uint32_t kLwRow = 4096;       // a row's albedo, normal, specular, mask keys
constexpr uint32_t kLwHashSize = 2048;  // the winners: tri bits + 1 -> slot
constexpr uint32_t kLwHashKeys = 4224;
constexpr uint32_t kLwHashSlots = 6272;
constexpr uint32_t kLwSlots = 8320;  // a winner's slot, then the empty pixels'
// a slot: material word, lit base rgb, specular rgb, tangent, bitangent, normal
constexpr uint32_t kLitSlotWords = 16;
constexpr uint32_t kLitSlotEmpty = 1024;  // the empty pixels' slot
static_assert(4 * (kLwSlots + kLitSlotWords * (kLitSlotEmpty + 1)) <= kLightWorkBytes);

// LIGHT's table (float32, light.py light_table): per channel c at 16 c the sky's terms,
// ambient, the sun; then the sun's direction, exposure, exposure / pi, the background.
constexpr uint32_t kLightTableBytes = 256;
constexpr uint32_t kLtChannel = 16;
constexpr uint32_t kLtSky = 0;  // K0, z z, x x - y y, x y, x z, y z, x, y, z
constexpr uint32_t kLtSkyTerms = 9;
constexpr uint32_t kLtAmbient = 9;
constexpr uint32_t kLtSun = 10;
constexpr uint32_t kLtSunDirection = 48;
constexpr uint32_t kLtExposure = 51;
constexpr uint32_t kLtScale = 52;       // exposure / pi
constexpr uint32_t kLtBackground = 53;  // exposure x FILL's background rgb
constexpr uint32_t kLtWords = 56;
// LIGHT's reader: an answer page an item, first_row | end_row << 8 (kShadeAskEnd: the
// end), and its planes: the surface's 0 to 4; the material word, the surface's 5, 6.
constexpr uint32_t kLightItemPages = 10;

// An ask page: word 0 the pages to read (kShadeAskEnd: the command's end), then the
// texture pages, the page for slot i at word 1 + i.
constexpr uint32_t kShadeAskEnd = 0xFFFFFFFF;

// The profile page (64 words) of a core: the pack thread's counters.
constexpr uint32_t kShadeProfBytes = 256;
constexpr uint32_t kShadeProfTiles = 0;
constexpr uint32_t kShadeProfRecords = 1;
constexpr uint32_t kShadeProfRows = 2;    // record rows run
constexpr uint32_t kShadeProfPasses = 3;  // passes of rows, each its own pages
constexpr uint32_t kShadeProfPages = 4;   // texture pages read
constexpr uint32_t kShadeProfTexels = 5;  // texels fetched (pixels with a texture)
constexpr uint32_t kShadeProfWaitIn = 6;  // cycles: list pages and planes
constexpr uint32_t kShadeProfCopyIn = 7;  // planes into Dst
constexpr uint32_t kShadeProfDivide = 8;  // u, v, light per pixel
constexpr uint32_t kShadeProfRecord = 9;  // the records' rows
constexpr uint32_t kShadeProfScan = 10;   // pages found, texel addresses
constexpr uint32_t kShadeProfWaitPages = 11;
constexpr uint32_t kShadeProfGather = 12;  // texels and their linear values to Dst
constexpr uint32_t kShadeProfOut = 13;     // colour, planes to the writer
constexpr uint32_t kShadeProfTotal = 14;
// LIGHT's profile page: tiles, rows, then cycles waiting for planes, copying in, the
// vector unit's arithmetic, copying out, in all (the same 64 words).
constexpr uint32_t kLightProfTiles = 0;
constexpr uint32_t kLightProfRows = 1;
constexpr uint32_t kLightProfWait = 2;
constexpr uint32_t kLightProfCopyIn = 3;
constexpr uint32_t kLightProfMath = 4;
constexpr uint32_t kLightProfOut = 5;
constexpr uint32_t kLightProfTotal = 6;

// Gave-up places (the writer's word): a wait of the unpack or pack thread, a bad
// header or table, the reader's loop.
constexpr uint32_t kWhereShadeMail = 29;
constexpr uint32_t kWhereShadeFront = 30;
constexpr uint32_t kWhereShadeBad = 31;
constexpr uint32_t kWhereShadeReader = 32;
