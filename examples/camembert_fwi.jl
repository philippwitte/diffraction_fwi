
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, NLopt, JLD

# Load starting model
M = load("../data/camembert_model.jld")
n = M["n"]; d = M["d"]; o = M["o"]  # dimensions, grid spacing, origin of coordinate system
v = M["v"]  # true velocity
m = (1f0 ./ v).^2f0 # squared slowness
m0 = ones(Float32, n) * m[1]    # initial model
model0 = Model(n, d, o, m0)

# Load data
block1 = segy_read("../data/observed_reflection_data.segy")
#block1 = segy_read("../data/observed_transmission_data.segy")
#block1 = segy_read("../data/observed_diffraction_data.segy")
d_obs = judiVector(block1)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32, model0.n) * 1.4f0
vmax = ones(Float32, model0.n) * 5f0

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# setup wavelet
src_geometry = Geometry(block1; key="source")
f0 = 0.005  # 5 Hz wavelet
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], f0)
q = judiVector(src_geometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)
info = Info(prod(n), d_obs.nsrc, ntComp)
F = judiModeling(info, model0, src_geometry, d_obs.geometry)
J = judiJacobian(F', q)

############################### FWI ###########################################

# optimization parameters
batchsize = d_obs.nsrc
count = 0

# NLopt objective function
println("No.  ", "fval         ", "norm(gradient)")
function f!(x, grad)

    # Update model
    model0.m = convert(Array{Float32, 2}, reshape(x, model0.n))
   
    # Select batch and calculate gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i])

    # Reset gradient around sources/receivers to 0
    gradient = reshape(gradient, model0.n)
    gradient[:, 1:21] = 0f0; gradient[:, 181:200] = 0f0
    grad[1:end] = vec(gradient)
          
    global count; count += 1
    println(count, "    ", fval, "    ", norm(grad))
    return convert(Float64, fval)
end

# Optimization parameters
opt = Opt(:LD_LBFGS, prod(model0.n))
lower_bounds!(opt, mmin); upper_bounds!(opt, mmax)
min_objective!(opt, f!)
maxeval!(opt, 16)
(minf, minx, ret) = optimize(opt, vec(model0.m))




