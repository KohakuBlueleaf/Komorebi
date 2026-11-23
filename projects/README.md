# Projects

Larger programs built on Komorebi, each in its own folder with its own README: what it
computes, how to run it, and what it is measured against. A project is where the model
is tried on real work; what it needs from the framework goes back into the framework
rather than staying in the project.

- [raster](raster/README.md): the rasterizer on the card, every core filling its tiles
  from tile lists bit for bit as the reference, with times, power and the V100 beside.
- [raster/reference](raster/reference/README.md): rasterization in numpy, threaded C++
  and CUDA, with the scenes, timings and pictures the card versions are compared with.
- [raytrace](raytrace/README.md): the ray tracer's whole frame on the card, bit for bit
  as numpy, with its times on the generated scenes and on the public Sponza and Bistro
  scenes.
- [raytrace/reference](raytrace/reference/README.md): ray tracing over a saved box tree
  in numpy, threaded C++ and CUDA, with the treelet cuts, timings and pictures.
- [viewer](viewer/README.md): a web page that steers the rasterizer's and the ray
  tracer's interactive loops: a browser moves the camera and picks the renderer and the
  scene, the card renders, the server streams the frames.
- [dense](dense/README.md): the matrix product on the persistent kernels, checked against
  float64 and measured against Kohaku-TT's routes on the same card.
- [transport](transport/README.md): a light-transport matrix stored as a hierarchical
  (block low-rank) matrix, applied on all 120 cores as one command, each core's blocks
  told by a table of entries, checked against the host's emulation of the same numbers.
- [upscale](upscale/README.md): a small super-resolution network (seven 3 x 3
  convolutions, 32 channels) taking a 512 x 512 frame to 1,024 x 1,024 as one chain
  command, each convolution nine tile products, checked against the host's emulation.
- [attention](attention/README.md): scaled dot-product attention on broadcasts of cores,
  K / V multicast down each broadcast, Kohaku-TT's flash compute body, checked against float64.
- [decode](decode/README.md): an LLM decode step, Qwen3-1.7B's 28 layers on the K split
  with the operations as overlays, walked by an X280 thread from one command, with an
  account of every microsecond above the DRAM read.
- [nn](nn/README.md): kernel calls as a list in DRAM and one generic X280 thread that
  runs it (a closed list replayed, an open list appended while it runs), the decode
  layer through it beside the hand-written thread, and what a call costs each way.
