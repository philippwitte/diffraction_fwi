
using JUDI.TimeModeling, HDF5, SeisIO

# Load velocity model
n, d, o, v = read(h5open("../data/camembert_model_padded.h5","r"), "n", "d", "o", "v")  # dimensions, grid spacing, grid origin, velocity
n = (n[1], n[2]); d = (d[1], d[2]); o = (o[1], o[2])    # convert to tuples

# Set up model structure w/ squared slowness
model = Model(n, d, o, (1f0 ./ v).^2f0)

# Setup info and model structure
nsrc = 33	# number of sources

## Set up receiver geometry
nxrec = 181
xrec = linspace(100f0, 1900f0, nxrec)
yrec = 0f0
zrec = linspace(100f0, 100f0, nxrec)

# receiver sampling and recording time
timeR = 2400f0   # receiver recording time [ms]
dtR = 4f0   # receiver sampling interval

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(linspace(200f0, 1800f0, nsrc))
ysrc = convertToCell(linspace(0f0, 0f0, nsrc))
zsrc = convertToCell(linspace(50f0, 50f0, nsrc))

# Source geometry for transmission data
#xsrc = convertToCell([250f0, 500f0, 750f0, 1000f0, 1250f0, 1500f0, 1750f0])
#ysrc = convertToCell([0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 0f0])
#zsrc = convertToCell([1600f0, 1600f0, 1600f0, 1600f0, 1600f0, 1600f0, 1600f0])

# source sampling and number of time steps
timeS = 2400f0
dtS = 4f0

# Set up source structure
src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.006
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(src_geometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(src_geometry, rec_geometry, model)
info = Info(prod(n), nsrc, ntComp)
F = judiModeling(info, model, src_geometry, rec_geometry)

# Model data
d_obs = F*q

# Convert to SeisIO blocks
block = judiVector_to_SeisBlock(d_obs, q)

# Write SEG-Y file
segy_write("../data/observed_data.segy", block)

