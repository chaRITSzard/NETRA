include("data.jl")
include("model.jl")

using Random
using MLJ
using Statistics

const RANDOM_SEED = 42
const VALIDATION_FRACTION = 0.20

println("Loading Training Data...")
X, y, feature_names = load_training_data()

println("Creating Validation Split...")

Random.seed!(RANDOM_SEED)

n=length(y)
indices=randperm(n)

n_validation = round(Int, VALIDATION_FRACTION * n)

validation_idx = indices[1:n_validation]
training_idx = indices[n_validation+1:end]

X_train = X[training_idx, :]
y_train = y[training_idx]

X_valid = X[validation_idx, :]
y_valid = y[validation_idx]

println("Training Rows: ", length(y_train))
println("Validation Rows: ", length(y_valid))


println("Building Random Forest...")

model = build_model(
    n_trees = 200,
    max_depth = -1,
    min_samples_leaf = 1,
    sampling_fraction = 0.7,
    n_subfeatures = 44.0,
    rng = 42,
)

println("Creating MLJ Machine...")
mach = build_machine(model, X_train, y_train)

println("Training model...")

training_start = time()
MLJ.fit!(mach, verbosity = 1)
training_time = time() - training_start

println("Generating validation predictions...")

prediction_start = time()
predictions = MLJ.predict(mach, Tables.table(X_valid))
prediction_time = time() - prediction_start

predicted_labels = String.(mode.(predictions))

accuracy = mean(predicted_labels .== y_valid)

println()
println("========== NETRA Julia Model ==========")
println("Model: RandomForestClassifier")
println("Trees: 200")
println("Max depth: -1")
println("Min samples leaf: 1")
println("Sampling fraction: 0.7")
println("Features per split: 44")
println("Training time: ", round(training_time, digits=4), " s")
println("Prediction time: ", round(prediction_time, digits=4), " s")
println("Validation accuracy: ", round(accuracy * 100, digits=4), "%")
println("======================================")