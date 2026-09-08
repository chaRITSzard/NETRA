using CSV
using DataFrames
using JSON

const DATA_DIR = joinpath(@__DIR__, "..", "benchmarks", "data")

function load_training_data()
    x_path = joinpath(DATA_DIR, "X_train_processed.csv")
    y_path = joinpath(DATA_DIR, "y_train.csv")
    features_path = joinpath(DATA_DIR, "feature_names.json")

    X = CSV.read(x_path, DataFrame)
    y_df = CSV.read(y_path, DataFrame)

    feature_names = String.(JSON.parsefile(features_path))

    @assert nrow(X) == 125973 "Unexpected number of training rows: $(nrow(X))"
    @assert ncol(X) == 44 "Unexpected number of processed features: $(ncol(X))"
    @assert nrow(y_df) == 125973 "Unexpected number of labels: $(nrow(y_df))"
    @assert length(feature_names) == 44 "Unexpected feature-name count: $(length(feature_names))"

    actual_names = String.(names(X))

    @assert actual_names == feature_names "Feature ordering mismatch"

    for column in names(X)
        @assert all(.!ismissing.(X[!, column])) "Missing values detected in $(column)"
    end

    y = String.(y_df[!, :attack_class])

    expected_classes = Set(["normal", "DoS", "Probe", "R2L", "U2R"])
    actual_classes = Set(y)

    @assert actual_classes == expected_classes "Unexpected attack classes: $(actual_classes)"

    X_matrix = Matrix{Float64}(X)

    @assert all(isfinite, X_matrix) "NaN or Inf detected in processed features"

    return X_matrix, y, feature_names
end