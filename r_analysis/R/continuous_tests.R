# Continuous feature statistical tests


source("r_analysis/R/labels.R")

numeric_features <- NSL_KDD_FEATURES[
  sapply(train_data[NSL_KDD_FEATURES], is.numeric)
]

run_continuous_tests <- function(data, feature) {

  formula <- as.formula(
    paste(feature, "~ attack_class")
  )

  # ANOVA: A statistical model used to compare the means of three or more groups by splitting 
  # overall data variation into between-group and within-group components.
  anova_model <- aov(formula, data = data)
  anova_result <- summary(anova_model)[[1]]

  anova_p <- anova_result$`Pr(>F)`[1]
  anova_f <- anova_result$`F value`[1]

  # Kruskal-Wallis: A rank-based non-parametric statistical test used to determine if there 
  # are statistically significant differences between three or more independent groups
  kruskal_result <- kruskal.test(
    formula,
    data = data
  )

  data.frame(
    feature = feature,
    anova_f = anova_f,
    anova_p_value = anova_p,
    kruskal_chi_square = as.numeric(
      kruskal_result$statistic
    ),
    kruskal_p_value = kruskal_result$p.value
  )
}

results <- do.call(
  rbind,
  lapply(
    numeric_features,
    function(feature) {
      run_continuous_tests(train_data, feature)
    }
  )
)

# Multiple-testing correction
results$anova_p_adjusted <- p.adjust(
  results$anova_p_value,
  method = "BH"
)

results$kruskal_p_adjusted <- p.adjust(
  results$kruskal_p_value,
  method = "BH"
)

cat("\n==============================\n")
cat("CONTINUOUS FEATURE TESTS\n")
cat("==============================\n\n")

print(
  results,
  row.names = FALSE
)

cat("\nSignificant features after BH correction:\n")

significant <- results[
  !is.na(results$kruskal_p_adjusted) &
  results$kruskal_p_adjusted < 0.05,
  c(
    "feature",
    "anova_p_adjusted",
    "kruskal_p_adjusted"
  )
]

print(
  significant,
  row.names = FALSE
)