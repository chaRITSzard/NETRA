# Chi-square tests for categorical features

source("r_analysis/R/labels.R")

categorical_features <- c(
  "protocol_type",
  "service",
  "flag"
)

run_chi_square <- function(data, feature) {

  contingency_table <- table(
    data[[feature]],
    data$attack_class
  )

  test <- chisq.test(contingency_table)

  # Cramer's V
  #0 means there is no association between the categorical variables.
  #1 means there is a perfect association between them.
  n <- sum(contingency_table)
  min_dim <- min(
    nrow(contingency_table) - 1,
    ncol(contingency_table) - 1
  )

  cramers_v <- sqrt(
    as.numeric(test$statistic) /
      (n * min_dim)
  )

  data.frame(
    feature = feature,
    chi_square = as.numeric(test$statistic),
    df = as.numeric(test$parameter),
    p_value = test$p.value,
    cramers_v = cramers_v
  )
}

results <- do.call(
  rbind,
  lapply(
    categorical_features,
    function(feature) {
      run_chi_square(train_data, feature)
    }
  )
)

cat("\n==============================\n")
cat("CHI-SQUARE ANALYSIS\n")
cat("==============================\n\n")

print(results, row.names = FALSE)