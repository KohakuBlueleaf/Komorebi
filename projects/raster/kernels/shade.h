// SHADE, between FILL and SHOW on a shaded frame: each core colours its tiles in
// place from materials and textures (reference/raster.py rasterize_shaded).
#pragma once

#include <stdint.h>

#include "grid.h"

// SHADE: a0 directory, a1 lists, a2 texture pages, a3 table, a4 profile, a6 the
// opaque directory (else 0; a0, a1 the masked or glass layer's), a9 kShade* bits.
constexpr uint32_t kHandlerShade = 5;  // lit: a7 the surface directory, a5 camera, a8 bank
// LIGHT: a0 directory, a7 surface, a5 camera (a8 bank), a1 table, a9 kLight*;
// kLightGlass: a6 the glass directory, a2 SHADE's table, a7 the glass surface.
constexpr uint32_t kHandlerLight = 6;
// SHADE's a9 (unlit merging): kShadeMergedDepth also the merged 1/z (plane 0);
// kShadeGlass: tau C + (1 - max tau) colour where nearer (a tie to the surface).
constexpr uint32_t kShadeMergedDepth = 1;
constexpr uint32_t kShadeGlass = 2;
// LIGHT's a9: 0 displayed; kLightLinear C (no exposure, no tone curve: glass next);
// kLightGlass the glass over C, then exposure and the tone curve everywhere.
constexpr uint32_t kLightLinear = 1;
constexpr uint32_t kLightGlass = 2;
// With a6, a pixel nearer than the opaque layer's (a tie to it) with alpha byte >=
// kAlphaOpaque takes the masked colour, else the opaque one, in the opaque frame.

// Buffers: the reader's planes, the table, the pack thread's page requests, the
// reader's answers, the page slots (plain memory), the pack thread's own space.
constexpr uint32_t kCbShadePlanes = 14;
constexpr uint32_t kShadePlanePages = 20;  // items of 5, 10 (a6), 15 (LIGHT) pages
constexpr uint32_t kCbShadeTable = 15;
// the table in L1: 816 + 14 per material + 2 per (texture, level) words; Bistro
// exterior's needs 7,254 (133 materials, 208 textures of 11 levels)
constexpr uint32_t kShadeTableBytes = 32768;
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
constexpr uint32_t kMaterialWords = 18;
constexpr uint32_t kMaterialMask = 4;      // 1 + mask's entry (sizes the albedo's)
constexpr uint32_t kMaterialNormal = 5;    // 1 + normal's entry
constexpr uint32_t kMaterialSpecular = 6;  // 1 + specular's entry
constexpr uint32_t kMaterialLitBase = 7;   // rgb, emissive: x emissive_scale
constexpr uint32_t kMaterialSpecularColour = 10;  // rgb, 0: no specular term
constexpr uint32_t kMaterialExponent = 13;        // lit: specular exponent (0: none)
constexpr uint32_t kMaterialTau = 14;     // transparent: transmission rgb (else 0)
constexpr uint32_t kMaterialWeight = 17;  // transparent: 1 - max tau (else 1)
constexpr uint32_t kEntryWords = 2;
// kinds (reference/raster.py): lit is shaded as textured in this step; masked as
// textured, its holes by the masked layer's SHADE (a6)
constexpr uint32_t kKindFlat = 0;
constexpr uint32_t kKindTextured = 1;
constexpr uint32_t kKindMasked = 2;
constexpr uint32_t kKindLit = 3;
constexpr uint32_t kKindEmissive = 4;
constexpr uint32_t kKindTransparent = 5;  // the glass layer (shaded as textured)
constexpr uint32_t kAlphaOpaque = 128;    // a masked texel under it is a hole

// Record word (SETUP writes it on a shaded frame): material | albedo level << 16 |
// normal level << 20 | specular level << 24.
constexpr uint32_t kRecShade = 30;
constexpr uint32_t kShadeLevelShift = 16;
constexpr uint32_t kShadeNormalLevelShift = 20;
constexpr uint32_t kShadeSpecularLevelShift = 24;
// lit: the triangle's coordinates give no tangent frame (the reference's frame_ok
// false, set by the host): its pixels take the triangle's normal
constexpr uint32_t kShadeNoFrame = 0x10000000;

// Lit SHADE's frame planes: 1/z (merged), material | kind << 16 | exponent << 20 in the
// triangle's plane (empty: FILL's -1, bit 31 set); planes 1 to 3 are LIGHT's.
constexpr uint32_t kMaterialNone = 0x80000000;
constexpr uint32_t kMaterialKindShift = 16;
constexpr uint32_t kMaterialKindMask = 0xF;
constexpr uint32_t kMaterialExponentShift = 20;
constexpr uint32_t kExponentMax = 2047;  // 11 bits under bit 31
// Its surface planes, two frames like the fill's: a (albedo x base, linear) rgb, normal
// x, y | normal z, visibility, specular rgb (x its texel; 0: no specular term).
constexpr uint32_t kSurfacePlanes = 10;
// The surface's directory page: the first frame's words at kDirFrame, the second's
// here.
constexpr uint32_t kSurfaceSecond = 0;
constexpr uint32_t kSurfaceWords = 32;  // both frames' words (kDirFrameMax + 1 each)
constexpr uint32_t kSurfaceLightTable = 16;    // LIGHT's table tensor
constexpr uint32_t kSurfaceLightProfile = 17;  // LIGHT's profile tensor (0: none)
constexpr uint32_t kSurfaceLightPages = 18;    // LIGHT's pages: shadow map, background
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

// LIGHT's table (light.py light_table; float32 unless int): per channel c at 16 c the
// sky's terms, ambient, the sun; then the words below, the camera (a5), each lamp.
constexpr uint32_t kLightTableBytes = 4096;
constexpr uint32_t kLtChannel = 16;
constexpr uint32_t kLtSky = 0;  // K0, z z, x x - y y, x y, x z, y z, x, y, z
constexpr uint32_t kLtSkyTerms = 9;
constexpr uint32_t kLtAmbient = 9;
constexpr uint32_t kLtSun = 10;
constexpr uint32_t kLtSunDirection = 48;
constexpr uint32_t kLtExposure = 51;
constexpr uint32_t kLtScale = 52;       // exposure / pi
constexpr uint32_t kLtBackground = 53;  // FILL's background rgb
constexpr uint32_t kLtInvPi = 56;
constexpr uint32_t kLtTerms = 57;         // int: kTerm* bits
constexpr uint32_t kLtExponentBits = 58;  // int: bits of the largest exponent
constexpr uint32_t kLtLamps = 59;         // int: lamps
constexpr uint32_t kLtShadowTexel = 60;   // a shadow texel's world size
constexpr uint32_t kLtShadowBias = 61;    // as 1/z
// the background's turn: cosine and sine of -turn (a view direction turned about y)
constexpr uint32_t kLtTurnCos = 62;
constexpr uint32_t kLtTurnSin = 63;
constexpr uint32_t kLtShadowCamera = 64;  // 16 words (reference/scenes.py)
// a map's pages from the light pages' page first - 1, across a row, side in texels
constexpr uint32_t kLtShadowFirst = 80;
constexpr uint32_t kLtShadowAcross = 81;
constexpr uint32_t kLtShadowSide = 82;
constexpr uint32_t kLtOctaFirst = 83;
constexpr uint32_t kLtOctaAcross = 84;
constexpr uint32_t kLtOctaSide = 85;
constexpr uint32_t kLtOctaLast = 86;     // side - 1
constexpr uint32_t kLtOctaChannel = 87;  // int: a channel's key step (pages << 10)
constexpr uint32_t kLtPages = 88;        // int: the light pages
constexpr uint32_t kLtReflection = 89;   // the glass's reflection of the environment
constexpr uint32_t kLtFresnel0 = 90;     // the glass's F0 (0.04)
constexpr uint32_t kLtFresnelRest = 91;  // 1 - F0
// the tone step (reference/raster.py tone): saturation (1: none), curve (int)
constexpr uint32_t kLtSaturation = 92;
constexpr uint32_t kLtCurve = 93;
constexpr uint32_t kCurvePlain = 0;   // x / (1 + x)
constexpr uint32_t kCurveFilmic = 1;  // x (2.51 x + 0.03) / (x (2.43 x + 0.59) + 0.14)
constexpr uint32_t kLtCamera = 96;    // 16 words, the reader's copy (a5)
constexpr uint32_t kLtLamp = 128;
// a lamp: position, colour, intensity, radius, 1 / radius, size^2
constexpr uint32_t kLampWords = 16;
constexpr uint32_t kLampColour = 3;
constexpr uint32_t kLampIntensity = 6;
constexpr uint32_t kLampRadius = 7;
constexpr uint32_t kLampInvRadius = 8;
constexpr uint32_t kLampSize2 = 9;
constexpr uint32_t kLampsMax = 32;
static_assert(4 * (kLtLamp + kLampWords * kLampsMax) <= kLightTableBytes);
// Indirect light (kTermIndirect; light.py field_words), floats: the field's grid, its
// pages (blocks across x, a y layer's blocks, 1 + first light page), the decode
constexpr uint32_t kLtFieldOrigin = 640;
constexpr uint32_t kLtFieldShift = 643;    // h / 2
constexpr uint32_t kLtFieldSpacing = 644;  // h
constexpr uint32_t kLtFieldInverse = 645;  // 1 / h
constexpr uint32_t kLtFieldDims = 646;     // voxels a side
constexpr uint32_t kLtFieldAcross = 649;
constexpr uint32_t kLtFieldLayer = 650;
constexpr uint32_t kLtFieldFirst = 651;
constexpr uint32_t kLtFieldUnit = 652;   // a channel: (1 / max code)^2 x the scale
constexpr uint32_t kLtFieldScale = 655;  // the scale (the host's record)
constexpr uint32_t kLtFieldPages = 656;  // int: the field's pages
static_assert(kLtFieldOrigin >= kLtLamp + kLampWords * kLampsMax);
static_assert(4 * (kLtFieldPages + 1) <= kLightTableBytes);
// A field page: 8 x 2 x 8 voxels (x, y, z), texel d 128 + (y 8 + z) 8 + x, d +x -x +y
// -y +z -z; blocks in x, z, y. Texel: bit 31 data, r 20-30, g 9-19, b 0-8 (pack_field).
constexpr uint32_t kFieldBlockX = 8;
constexpr uint32_t kFieldBlockY = 2;
constexpr uint32_t kFieldBlockZ = 8;
constexpr uint32_t kFieldDirectionTexels = 128;
static_assert(kFieldBlockX * kFieldBlockY * kFieldBlockZ == kFieldDirectionTexels);
static_assert(6 * kFieldDirectionTexels * 4 <= kTexPageBytes);
// The terms past part A's (sky, ambient, the sun): its shadow, the lamps, the specular
// term, the background map (without: FILL's background), the indirect light (field).
constexpr uint32_t kTermShadow = 1;
constexpr uint32_t kTermLamps = 2;
constexpr uint32_t kTermSpecular = 4;
constexpr uint32_t kTermBackground = 8;
constexpr uint32_t kTermIndirect = 16;
// LIGHT's reader: an item's word in kCbIn (first_row | end_row << 8, kShadeAskEnd: the
// end; then its tile) and its planes: the surface's ten, the frame's 1/z, material word
constexpr uint32_t kLightItemPlanes = 12;
// in three parts of five pages: the ring's place stays a multiple of five for SHADE
constexpr uint32_t kLightItemPages = 15;
static_assert(kLightItemPages % 5 == 0 && kShadePlanePages % 5 == 0);
static_assert(
    kLightItemPlanes <= kLightItemPages && kLightItemPages <= kShadePlanePages
);
// kLightGlass: the glass layer's twelve planes, the opaque frame's C and merged 1/z,
// in four parts of five pages (the ring's whole size)
constexpr uint32_t kLightGlassPlanes = 16;
constexpr uint32_t kLightGlassC = 12;
constexpr uint32_t kLightGlassDepth = 15;
constexpr uint32_t kLightGlassPages = 20;
static_assert(kLightGlassPages % 5 == 0 && kLightGlassPages == kShadePlanePages);
static_assert(kLightGlassPlanes <= kLightGlassPages);
// kLightGlass: SHADE's table after LIGHT's in kCbShadeTable (its first words: counts,
// the texel tables and the materials, which must end before the buffer's end)
constexpr uint32_t kLightShadeTableAt = kLightTableBytes / 4;
constexpr uint32_t kLightShadeTableBytes = kShadeTableBytes - kLightTableBytes;

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
// not summed: the most pages one item's rows read, the items read in more than one
// pass
constexpr uint32_t kShadeProfItemPages = 15;
constexpr uint32_t kShadeProfItemPasses = 16;
constexpr uint32_t kShadeProfWords = 17;
// LIGHT's profile page: tiles, rows, then cycles waiting for planes, copying in, the
// vector unit's arithmetic, copying out, the texel steps; pages, passes; in all.
constexpr uint32_t kLightProfTiles = 0;
constexpr uint32_t kLightProfRows = 1;
constexpr uint32_t kLightProfWait = 2;
constexpr uint32_t kLightProfCopyIn = 3;
constexpr uint32_t kLightProfMath = 4;  // the colour rows but the three below
constexpr uint32_t kLightProfOut = 5;
constexpr uint32_t kLightProfKeys = 6;  // the texel keys (shadow, background)
constexpr uint32_t kLightProfScan = 7;  // pages found, texel addresses
constexpr uint32_t kLightProfWaitPages = 8;
constexpr uint32_t kLightProfGather = 9;     // texels to Dst
constexpr uint32_t kLightProfShadow = 10;    // of kLightProfMath: the shadow's filter
constexpr uint32_t kLightProfLamps = 11;     // the lamps (their specular terms too)
constexpr uint32_t kLightProfSpecular = 12;  // the sun's specular term
constexpr uint32_t kLightProfPages = 13;
constexpr uint32_t kLightProfPasses = 14;
constexpr uint32_t kLightProfTotal = 15;
// not in the total's parts: kLightGlass's quarter tiles with glass in front (the
// others only toned)
constexpr uint32_t kLightProfGlassQuarters = 16;
// the field (kTermIndirect), parts of the total: keys, scan, wait, gather (texels to
// Dst), blend; then its pages, passes and the most pages one item read (not summed)
constexpr uint32_t kLightProfFieldKeys = 17;
constexpr uint32_t kLightProfFieldScan = 18;
constexpr uint32_t kLightProfFieldWait = 19;
constexpr uint32_t kLightProfFieldGather = 20;
constexpr uint32_t kLightProfFieldBlend = 21;
constexpr uint32_t kLightProfFieldPages = 22;
constexpr uint32_t kLightProfFieldPasses = 23;
constexpr uint32_t kLightProfFieldItemPages = 24;
constexpr uint32_t kLightProfWords = 25;

// Gave-up places (the writer's word): a wait of the unpack or pack thread, a bad
// header or table, the reader's loop.
constexpr uint32_t kWhereShadeMail = 29;
constexpr uint32_t kWhereShadeFront = 30;
constexpr uint32_t kWhereShadeBad = 31;
constexpr uint32_t kWhereShadeReader = 32;
