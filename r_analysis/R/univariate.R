# NSL-KDD univariate analysis

source("r_analysis/R/labels.R")

feature_data <- train_data[, NSL_KDD_FEATURES]

# Identify feature types
numeric_features <- names(feature_data)[
  sapply(feature_data, is.numeric)
]

categorical_features <- names(feature_data)[
  sapply(feature_data, is.factor)
]

cat("\n==============================\n")
cat("UNIVARIATE ANALYSIS\n")
cat("==============================\n")

cat("\nNumeric features:", length(numeric_features), "\n")
cat("Categorical features:", length(categorical_features), "\n")

# Numeric summaries
numeric_summary <- data.frame(
  feature = numeric_features,
  mean = sapply(
    feature_data[numeric_features],
    mean,
    na.rm = TRUE
  ),
  sd = sapply(
    feature_data[numeric_features],
    sd,
    na.rm = TRUE
  ),
  median = sapply(
    feature_data[numeric_features],
    median,
    na.rm = TRUE
  ),
  q1 = sapply(
    feature_data[numeric_features],
    quantile,
    probs = 0.25,
    na.rm = TRUE
  ),
  q3 = sapply(
    feature_data[numeric_features],
    quantile,
    probs = 0.75,
    na.rm = TRUE
  ),
  min = sapply(
    feature_data[numeric_features],
    min,
    na.rm = TRUE
  ),
  max = sapply(
    feature_data[numeric_features],
    max,
    na.rm = TRUE
  )
)

cat("\nNumeric feature summaries:\n")
print(numeric_summary, row.names = FALSE)

# Categorical summaries
cat("\nCategorical feature summaries:\n")

for (feature in categorical_features) {

  cat("\n---", feature, "---\n")

  counts <- sort(
    table(feature_data[[feature]]),
    decreasing = TRUE
  )

  print(counts)
}