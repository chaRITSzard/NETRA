include("data.jl")
include("model.jl")

using MLJ
using Tables
using BenchmarkTools
using Statistics
using Random
using CSV
using DataFrames
using Printf

const RANDOM_SEED = 42
const VALIDATION_FRACTION = 0.20
const RESULTS_DIR = joinpath(@__DIR__, "..", "reports")
const RESULTS_PATH = joinpath(RESULTS_DIR, "benchmark_results.csv")

function accuracy_score(y_true, y_pred)
    return mean(y_true .== y_pred)
end

function macro_f1_score(y_true, y_pred)
    classes = ["normal", "DoS", "Probe", "R2L", "U2R"]
    f1_scores = Float64[]

    for class in classes
        tp = sum((y_true .== class) .& (y_pred .== class))
        fp = sum((y_true .!= class) .& (y_pred .== class))
        fn = sum((y_true .== class) .& (y_pred .!= class))

        precision = tp + fp == 0 ? 0.0 : tp / (tp + fp)
        recall = tp + fn == 0 ? 0.0 : tp / (tp + fn)

        f1 = precision + recall == 0 ?
            0.0 :
            2 * precision * recall / (precision + recall)

        push!(f1_scores, f1)
    end

    return mean(f1_scores)
end

function create_split(X, y)
    Random.seed!(RANDOM_SEED)

    n = length(y)
    indices = randperm(n)
    n_validation = round(Int, VALIDATION_FRACTION * n)

    validation_idx = indices[1:n_validation]
    training_idx = indices[n_validation+1:end]

    return (
        X[training_idx, :],
        y[training_idx],
        X[validation_idx, :],
        y[validation_idx],
    )
end

function benchmark_configuration(
    X_train,
    y_train,
    X_valid,
    y_valid,
    ;
    n_trees,
    max_depth,
    min_samples_leaf,
    n_subfeatures,
)
    model = build_model(
        n_trees = n_trees,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        sampling_fraction = 0.7,
        n_subfeatures = Float64(n_subfeatures),
        rng = RANDOM_SEED,
    )

    mach = build_machine(model, X_train, y_train)

    println()
    println(
        "Benchmarking: ",
        "trees=", n_trees,
        ", depth=", max_depth,
        ", leaf=", min_samples_leaf,
        ", features=", n_subfeatures
    )

    GC.gc()

    training_result = @timed begin
        MLJ.fit!(mach, verbosity = 0)
    end

    GC.gc()

    prediction_result = @timed begin
        predictions = MLJ.predict(
            mach,
            Tables.table(X_valid)
        )
        String.(mode.(predictions))
    end

    predicted_labels = prediction_result.value

    accuracy = accuracy_score(y_valid, predicted_labels)
    macro_f1 = macro_f1_score(y_valid, predicted_labels)

    println(
        @sprintf(
            "  train: %.4f s | predict: %.4f s | accuracy: %.4f%% | macro-F1: %.6f",
            training_result.time,
            prediction_result.time,
            accuracy * 100,
            macro_f1,
        )
    )

    return (
        n_trees = n_trees,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        n_subfeatures = n_subfeatures,
        sampling_fraction = 0.7,
        training_seconds = training_result.time,
        prediction_seconds = prediction_result.time,
        training_allocated_bytes = training_result.bytes,
        prediction_allocated_bytes = prediction_result.bytes,
        accuracy = accuracy,
        macro_f1 = macro_f1,
    )
end


println("Loading NETRA training data...")

X, y, feature_names = load_training_data()

X_train, y_train, X_valid, y_valid = create_split(X, y)

println("Training rows: ", length(y_train))
println("Validation rows: ", length(y_valid))

configs = [
    (100, 10, 1, 11),
    (100, 10, 1, 22),
    (100, 10, 1, 44),
    (100, 20, 1, 11),
    (100, 20, 1, 22),

    (200, 10, 1, 11),
    (200, 10, 1, 22),
    (200, 10, 1, 44),
    (200, 20, 1, 11),
    (200, 20, 1, 22),

    (200, -1, 1, 11),
    (200, -1, 1, 22),
    (200, -1, 1, 44),

    (400, 10, 1, 22),
    (400, 20, 1, 22),
    (400, -1, 1, 22),

    (400, 20, 2, 22),
    (400, -1, 2, 22),

    (600, 20, 1, 22),
    (600, -1, 1, 22),
]

mkpath(RESULTS_DIR)

results = NamedTuple[]

println()
println("========================================")
println("NETRA Julia Benchmark")
println("Configurations: ", length(configs))
println("========================================")

for (i, config) in enumerate(configs)
    println()
    println("Configuration ", i, "/", length(configs))

    result = benchmark_configuration(
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_trees = config[1],
        max_depth = config[2],
        min_samples_leaf = config[3],
        n_subfeatures = config[4],
    )

    push!(results, result)
end

results_df = DataFrame(results)

sort!(results_df, :macro_f1, rev = true)

CSV.write(RESULTS_PATH, results_df)

println()
println("========================================")
println("Benchmark complete")
println("Results: ", RESULTS_PATH)
println("========================================")

println()
println("Top configurations by macro-F1:")

for row in eachrow(first(results_df, min(5, nrow(results_df))))
    println(
        @sprintf(
            "trees=%d depth=%d leaf=%d features=%d | accuracy=%.4f%% | macro-F1=%.6f | train=%.4fs",
            row.n_trees,
            row.max_depth,
            row.min_samples_leaf,
            row.n_subfeatures,
            row.accuracy * 100,
            row.macro_f1,
            row.training_seconds,
        )
    )
end