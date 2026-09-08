include("data.jl")
include("model.jl")
using MLJ

println("Loading Feature Data...")
X, y, features = load_training_data()

println("Building Random Forest...")
model = build_model(
    n_trees=200,
    max_depth=-1,
    min_samples_leaf=1,
    sampling_fraction=0.7,
    n_subfeatures=1.0,
    rng=42,
)

println("Creating MLJ machine...")
mach = build_machine(model, X, y)

println("Training...")
MLJ.fit!(mach, verbosity=1)

println("Training Complete")