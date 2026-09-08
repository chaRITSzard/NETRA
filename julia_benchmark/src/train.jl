include("data.jl")
include("model.jl")

using MLJ
using Tables
using Statistics
using Random
using Printf
using CSV
using DataFrames
using Serialization

const RANDOM_SEED = 42
const VALIDATION_FRACTION = 0.20

const CLASS_ORDER = ["normal", "DoS", "Probe", "R2L", "U2R"]

const RESULTS_DIR = joinpath(@__DIR__, "..", "reports")
const MODEL_DIR = joinpath(@__DIR__, "..", "models")

const METRICS_PATH = joinpath(RESULTS_DIR, "final_model_metrics.csv")
const CONFUSION_PATH = joinpath(RESULTS_DIR, "final_model_confusion_matrix.csv")
const MODEL_PATH = joinpath(MODEL_DIR, "netra_random_forest.jls")


function classification_metrics(y_true, y_pred)

    confusion = zeros(Int, length(CLASS_ORDER), length(CLASS_ORDER))

    class_to_index = Dict(
        class => i for (i, class) in enumerate(CLASS_ORDER)
    )

    for i in eachindex(y_true)

        actual = class_to_index[y_true[i]]
        predicted = class_to_index[y_pred[i]]

        confusion[actual, predicted] += 1
    end

    precision = Dict{String, Float64}()
    recall = Dict{String, Float64}()
    f1 = Dict{String, Float64}()
    support = Dict{String, Int}()

    for (i, class) in enumerate(CLASS_ORDER)

        tp = confusion[i, i]

        actual_count = sum(confusion[i, :])
        predicted_count = sum(confusion[:, i])

        support[class] = actual_count

        precision[class] =
            predicted_count == 0 ?
            0.0 :
            tp / predicted_count

        recall[class] =
            actual_count == 0 ?
            0.0 :
            tp / actual_count

        f1[class] =
            precision[class] + recall[class] == 0 ?
            0.0 :
            2 * precision[class] * recall[class] /
            (precision[class] + recall[class])
    end

    accuracy = mean(y_true .== y_pred)

    macro_precision = mean(
        precision[class] for class in CLASS_ORDER
    )

    macro_recall = mean(
        recall[class] for class in CLASS_ORDER
    )

    macro_f1 = mean(
        f1[class] for class in CLASS_ORDER
    )

    total_support = sum(
        support[class] for class in CLASS_ORDER
    )

    weighted_f1 =
        sum(
            f1[class] * support[class]
            for class in CLASS_ORDER
        ) / total_support

    return (
        accuracy = accuracy,
        precision = precision,
        recall = recall,
        f1 = f1,
        support = support,
        macro_precision = macro_precision,
        macro_recall = macro_recall,
        macro_f1 = macro_f1,
        weighted_f1 = weighted_f1,
        confusion = confusion,
    )
end


mkpath(RESULTS_DIR)
mkpath(MODEL_DIR)

println("Loading NETRA training data...")

X, y, feature_names = load_training_data()

println("Creating fixed validation split...")

Random.seed!(RANDOM_SEED)

n = length(y)
indices = randperm(n)

n_validation = round(
    Int,
    VALIDATION_FRACTION * n
)

validation_idx = indices[1:n_validation]
training_idx = indices[n_validation+1:end]

X_train = X[training_idx, :]
y_train = y[training_idx]

X_valid = X[validation_idx, :]
y_valid = y[validation_idx]

println("Training rows: ", length(y_train))
println("Validation rows: ", length(y_valid))

println()
println("Building final Julia model...")

model = build_model(
    n_trees = 100,
    max_depth = 20,
    min_samples_leaf = 1,
    sampling_fraction = 0.7,
    n_subfeatures = 22.0,
    rng = RANDOM_SEED,
)

println("Creating MLJ machine...")

mach = build_machine(
    model,
    X_train,
    y_train,
)

println("Training final model...")

training_start = time()

MLJ.fit!(
    mach,
    verbosity = 1,
)

training_time = time() - training_start

println("Training complete.")

println("Generating validation predictions...")

prediction_start = time()

predictions = MLJ.predict(
    mach,
    Tables.table(X_valid),
)

prediction_time = time() - prediction_start

predicted_labels = String.(mode.(predictions))

metrics = classification_metrics(
    y_valid,
    predicted_labels,
)


println()
println("========== NETRA FINAL JULIA MODEL ==========")

println("Model: RandomForestClassifier")
println("Trees: 100")
println("Max depth: 20")
println("Min samples leaf: 1")
println("Sampling fraction: 0.7")
println("Features per split: 22")
println("Random seed: 42")

@printf(
    "Training time: %.4f s\n",
    training_time
)

@printf(
    "Prediction time: %.4f s\n",
    prediction_time
)

println()

@printf(
    "Accuracy: %.6f%%\n",
    metrics.accuracy * 100
)

@printf(
    "Macro precision: %.6f\n",
    metrics.macro_precision
)

@printf(
    "Macro recall: %.6f\n",
    metrics.macro_recall
)

@printf(
    "Macro F1: %.6f\n",
    metrics.macro_f1
)

@printf(
    "Weighted F1: %.6f\n",
    metrics.weighted_f1
)

println()
println("Per-class metrics:")

@printf(
    "%-10s %12s %12s %12s %10s\n",
    "Class",
    "Precision",
    "Recall",
    "F1",
    "Support"
)

for class in CLASS_ORDER

    @printf(
        "%-10s %12.6f %12.6f %12.6f %10d\n",
        class,
        metrics.precision[class],
        metrics.recall[class],
        metrics.f1[class],
        metrics.support[class],
    )

end


println()
println("Confusion matrix:")
println("Rows = actual, columns = predicted")
println()

@printf("%-10s", "")

for class in CLASS_ORDER
    @printf("%10s", class)
end

println()

for (i, class) in enumerate(CLASS_ORDER)

    @printf("%-10s", class)

    for j in 1:length(CLASS_ORDER)
        @printf(
            "%10d",
            metrics.confusion[i, j]
        )
    end

    println()
end


metrics_df = DataFrame(
    metric = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "training_seconds",
        "prediction_seconds",
    ],
    value = [
        metrics.accuracy,
        metrics.macro_precision,
        metrics.macro_recall,
        metrics.macro_f1,
        metrics.weighted_f1,
        training_time,
        prediction_time,
    ],
)

CSV.write(
    METRICS_PATH,
    metrics_df,
)


confusion_df = DataFrame(
    actual = CLASS_ORDER,
)

for (j, class) in enumerate(CLASS_ORDER)
    confusion_df[!, Symbol(class)] =
        metrics.confusion[:, j]
end

CSV.write(
    CONFUSION_PATH,
    confusion_df,
)


println()
println("Saving final Julia model...")

serialize(
    MODEL_PATH,
    mach,
)

println()
println("Model saved: ", MODEL_PATH)
println("Metrics saved: ", METRICS_PATH)
println("Confusion matrix saved: ", CONFUSION_PATH)

println()
println("=============================================")