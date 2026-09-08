using MLJ
using DecisionTree

const RF = @load RandomForestClassifier pkg=DecisionTree verbosity=0

function build_model(
    ;
    n_trees::Int=200,
    max_depth::Int=-1,
    min_samples_leaf::Int=1,
    sampling_fraction::Float64=0.7,
    n_subfeatures::Float64=1.0,
    rng::Int=42,
)
    return RF(
        n_trees=n_trees,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        sampling_fraction=sampling_fraction,
        n_subfeatures=n_subfeatures,
        rng=rng
    )
end

function build_machine(model, X, y)
    X_table = Tables.table(X)
    y_cat = categorical(y)

    return machine(model, X_table, y_cat)
end