# NSL-KDD data loading
# NETRA - Network Threat Recognition & Analysis

source("r_analysis/R/schema.R")

TRAIN_PATH <- "data/raw/KDDTrain+.txt"
TEST_PATH  <- "data/raw/KDDTest+.txt"

load_nsl_kdd <- function(path) {
  if (!file.exists(path)) {
    stop("Dataset not found: ", path)
  }

  data <- read.csv(
    path,
    header = FALSE,
    stringsAsFactors = FALSE
  )

  if (ncol(data) != length(NSL_KDD_COLUMNS)) {
    stop(
      "Expected ",
      length(NSL_KDD_COLUMNS),
      " columns, found ",
      ncol(data)
    )
  }

  colnames(data) <- NSL_KDD_COLUMNS

  data$protocol_type <- as.factor(data$protocol_type)
  data$service <- as.factor(data$service)
  data$flag <- as.factor(data$flag)
  data$label <- as.factor(data$label)

  data
}

train_data <- load_nsl_kdd(TRAIN_PATH)
test_data  <- load_nsl_kdd(TEST_PATH)

cat("Training rows:", nrow(train_data), "\n")
cat("Test rows:", nrow(test_data), "\n")
cat("Columns:", ncol(train_data), "\n")