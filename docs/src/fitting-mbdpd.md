# Fitting a Friction Tensor for Simulation of (Multi-Body) Dissipative Particle Dynamics 

In this workflow example we demonstrate how `ACEfriction.jl` can be used to fit a momentum-conserving friction tensor as used in Dissipative Particle Dynamics. 

## Background on Dissipative Particle Dynamics
Dissipative particle dynamics can be considered as a special version of the Langevin equation @ref, where the friction tensor $\Gamma$ is such that the total momentum is conserved, i.e.
```math
\frac{d}{dt}\sum_i p_i(t) = {\bf 0}.
```
In order for this to be the case, the friction tensor must satisfy the constraint
```math
\sum_{i}\Gamma_{ij} = {\bf 0}, \text{ for every } j=1,\dots, N_{\rm at}.
```

## Momentum-conserving Friction Models in `ACEfriction.jl`
`ACEfriction.jl` provides utility functions for the construction of momentum-conserving friction models. Namely, the function [mdDPD_pwc_matrixmodel]() yields a pair-wise coupled matrix model with additional symmetries such that resulting friction model satisfies the above constraints (and [mdDPD_ac_matrixmodel]() the atom-centred variant). For example, 
```julia
m_cov = mdDPD_pwc_matrixmodel(EuclideanVector(), [:X], [:X];
    maxorder=1, 
    maxdeg=5,    
    rcutbond = 5.0, 
    rcutenv = 5.0,
    zcutenv = 5.0,
    n_rep = 1, 
    )
fm= FrictionModel((m_cov=m_cov,)); 
```
results in a momentum-conserving friction model with vector-equivariant blocks in the diffusion matrix. Here, the model is specified for the artifical atom element type `:X`.

### Two momentum-conserving constructions

Momentum conservation requires $\sum_i {\bm \Gamma}_{ij} = {\bm 0}$ for every $j$ (so that a rigid translation exerts no net friction force). How this is secured depends on how ${\bm \Gamma}$ is assembled from the diffusion matrix ${\bm \Sigma}$ — which differs between the coupling schemes (see [Assembly of the friction tensor](@ref gamma-assembly)). `ACEfriction.jl` offers two constructions:

1. **Pair-wise coupling** — [`mdDPD_pwc_matrixmodel`](@ref) builds a [`PWCMatrixModel`](@ref) with a purely off-diagonal, *antisymmetric* diffusion matrix,
   ```math
   {\bm \Sigma}_{ij} = -{\bm \Sigma}_{ji}, \qquad {\bm \Sigma}_{ii} = {\bm 0}.
   ```
   Under the pair-wise assembly ${\bm \Gamma}_{ij} = {\bm \Sigma}_{ij}({\bm \Sigma}_{ji})^{T}$ ($i\neq j$), ${\bm \Gamma}_{ii} = \sum_k {\bm \Sigma}_{ik}({\bm \Sigma}_{ik})^{T}$, antisymmetry gives ${\bm \Gamma}_{ij} = -{\bm \Sigma}_{ij}({\bm \Sigma}_{ij})^{T}$ for $i\neq j$, hence $\sum_i {\bm \Gamma}_{ij} = {\bm \Gamma}_{jj} - \sum_{i\neq j}{\bm \Sigma}_{ij}({\bm \Sigma}_{ij})^{T} = {\bm 0}$. The antisymmetry is enforced either by the **Z2-odd parity** of the bond basis on a bond-centred `EllipsoidCutoff` (`env=:ellipsoid`, the default), or by an **antisymmetric snowman combine** on a `SnowManCutoff` (`env=:snowman`):
   ```math
   {\bm \Sigma}_{ij} = c\cdot B(\text{sphere}_i, i\to j) - c\cdot B(\text{sphere}_j, j\to i).
   ```

2. **Atom-centred coupling** — [`mdDPD_ac_matrixmodel`](@ref) builds an [`ACDPDMatrixModel`](@ref) on a `SphericalCutoff` whose off-diagonal blocks are ordinary offsite ACE blocks and whose diagonal is *derived* so that ${\bm \Sigma}$ has **vanishing column sums**, $\sum_i {\bm \Sigma}_{ij} = {\bm 0}$:
   ```math
   {\bm \Sigma}_{ij} = \begin{cases} -\sum_{k\neq i} {\bm \Sigma}_{ki}, & j = i,\\[2pt] {\bm \Sigma}_{ij}, & i\neq j. \end{cases}
   ```
   Here ${\bm \Gamma}={\bm \Sigma}{\bm \Sigma}^{T}$ (row-wise assembly), so vanishing column sums give $\sum_i {\bm \Gamma}_{ij} = {\bm 0}$ directly, and the random force ${\bm F}={\bm \Sigma}{\bm R}$ conserves momentum, $\sum_i {\bm F}_i = \sum_j\big(\sum_i {\bm \Sigma}_{ij}\big){\bm R}_j = {\bm 0}$. This includes the on-site (diagonal) friction contribution explicitly, at the cost of one column-sum pass over the off-diagonal blocks; the diagonal introduces no additional fit parameters.

For example, the atom-centred variant is constructed as
```julia
m_cov = mdDPD_ac_matrixmodel(EuclideanVector(), [:X], [:X];
    maxorder = 1, maxdeg = 5, rcut = 5.0, n_rep = 1)
```

## Fit Friction Model to Synthetic DPD Friction Data

The following code loads training and test data comprised of particle configurations and corresponding friction tensors:
```julia
rdata_train = ACEfriction.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 
rdata_test = ACEfriction.DataUtils.load_h5fdata("./examples/data/dpd-train-x.h5"); 

fdata = Dict("train" => rdata_train, 
            "test"=> rdata_test);
(n_train, n_test) = length(fdata["train"]), length(fdata["test"])
```
Here the training data is contains friction tensors of 50 configurations each comprised of 64 particles, and the test data contains friction tensors of 10 configurations each comprised of 216 particles. The underlying friction tensors were synthetically generated using the following simple friction model, which is a smooth version of the standard heuristic DPD friction models commonly used in simulations: 
```math
\Gamma_{ij}( ({\bf r}_k,z_k )_{k=1}^{N_{\rm at}}) := \begin{cases}
\gamma(r_{ij}) \,\hat{\bf r}_{ij} \otimes \hat{\bf r}_{ji}, &i \neq j, \\
-\sum_{k \neq i} \Gamma_{ki}, &i = j,
\end{cases}
```
where ${\bf r}_{ij} := {\bf r}_{j} - {\bf r}_{i}$,  $r_{ij} := \|{\bf r}_{ij}\|_2$, $\hat{\bf r}_{ij} := {\bf r}_{ij}/r_{ij}$, and
```math
\gamma(r) := w \exp \left (-\frac{1}{1-(r/r_{\rm cut})^2} \right ), 
```
with weight $w=5.0$ and cutoff distance $r_{\rm cut}=5.0$.

To fit the model we execute exactly the same steps as in the previous example:

```julia
ffm = FluxFrictionModel(params(fm;format=:matrix, joinsites=true))
flux_data = Dict( "train"=> flux_assemble(fdata["train"], fm, ffm),
                  "test"=> flux_assemble(fdata["test"], fm, ffm));


loss_traj = Dict("train"=>Float64[], "test" => Float64[])
epoch = 0
batchsize = 10
nepochs = 100
opt = Flux.setup(Adam(1E-2, (0.99, 0.999)),ffm)
dloader = DataLoader(flux_data["train"], batchsize=batchsize, shuffle=true)

for _ in 1:nepochs
    epoch+=1
    @time for d in dloader
        ∂L∂m = Flux.gradient(weighted_l2_loss,ffm, d)[1]
        Flux.update!(opt,ffm, ∂L∂m)       # method for "explicit" gradient
    end
    for tt in ["test","train"]
        push!(loss_traj[tt], weighted_l2_loss(ffm,flux_data[tt]))
    end
    println("Epoch: $epoch, Abs avg Training Loss: $(loss_traj["train"][end]/n_train)), Test Loss: $(loss_traj["test"][end]/n_test))")
end
```


After training for 2000 epochs, the resulting model is almost a perfect fit:

![True vs fitted entries of the friction tensor](./assets/scatter-equ-cov.jpg)
            True vs fitted entries of the friction tensor.

## Multi-Body Dissipative Particle Dynamics

By specifying `maxorder=1` in the above construction of the friction model, we restrict the underlying ACE-basis to only incorporate pair-wise interactions. This is fine for the here considered sythentic data as the underlying toy model is in fact based on only pair-wise interactions. However, in more complex systems  the random force and the dissipative force may not decompose to pairwise interactions. To incorporate higher body-order interactions in the friction model, say interactions up to body order 4, we can change the underlying ACE-basis expansion to incorporate correlation terms up to order 3 by setting `maxorder=3`. 

