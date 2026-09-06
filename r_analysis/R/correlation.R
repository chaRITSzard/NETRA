# Correlation analysis

source("r_analysis/R/labels.R")

numeric_features <- NSL_KDD_FEATURES[
  sapply(train_data[NSL_KDD_FEATURES], is.numeric)
]

numeric_data <- train_data[, numeric_features]

feature_sd <- sapply(numeric_data, sd, na.rm = TRUE)

zero_variance_features <- names(feature_sd[
  feature_sd == 0
])

if (length(zero_variance_features) > 0) {
  numeric_data <- numeric_data[
    ,
    !names(numeric_data) %in% zero_variance_features,
    drop = FALSE
  ]
}

cor_matrix <- cor(
  numeric_data,
  method = "pearson",
  use = "pairwise.complete.obs"
)

threshold <- 0.85

cor_pairs <- which(
  abs(cor_matrix) > threshold &
  upper.tri(cor_matrix),
  arr.ind = TRUE
)

cat("\n==============================\n")
cat("CORRELATION ANALYSIS\n")
cat("==============================\n\n")

cat(
  "Zero-variance features removed from correlation analysis:",
  length(zero_variance_features),
  "\n"
)

if (length(zero_variance_features) > 0) {
  print(zero_variance_features)
}

cat(
  "\nHighly correlated pairs (|r| >",
  threshold,
  "):\n"
)

if (nrow(cor_pairs) == 0) {

  cat("No highly correlated pairs found.\n")

} else {

  correlation_results <- data.frame(
    feature_1 = rownames(cor_matrix)[cor_pairs[, 1]],
    feature_2 = colnames(cor_matrix)[cor_pairs[, 2]],
    correlation = cor_matrix[cor_pairs]
  )

  correlation_results <- correlation_results[
    order(
      -abs(correlation_results$correlation)
    ),
  ]

  print(
    correlation_results,
    row.names = FALSE
  )
}