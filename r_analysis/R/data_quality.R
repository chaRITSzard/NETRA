#NSL-KDD Data Quality Checks

source("r_analysis/R/load_data.R")

check_data_quality <- function(data, dataset_name){
    cat("\n===============================\n")
    cat(dataset_name, "\n")
    cat("\n===============================\n")

    cat("Rows:", nrow(data), "\n")
    cat("Columns:", ncol(data), "\n")

    missing_data <- sum(is.na(data))
    cat("Missing Data:", missing_data, "\n")

    duplicate_rows <- sum(duplicated(data))
    cat("Duplicate Rows:", duplicate_rows, "\n")

    cat("\nData Types:\n")
    print(table(sapply(data, class)))

    cat("\nLabel Distribution\n")
    print(sort(table(data$label), decreasing = TRUE))

    cat("\nCategorical Cardinality:\n")
    cat("protocol_type:", nlevels(data$protocol_type), "\n")
    cat("service:", nlevels(data$service), "\n")
    cat("flag:", nlevels(data$flag), "\n")

    invisible(list(
        rows = nrow(data),
        columns = ncol(data),
        missing = missing_data,
        duplicates = duplicate_rows
    ))
}

train_quality <- check_data_quality(
    train_data,
    "KDDTrain+"
)

test_quality <- check_data_quality(
    test_data,
    "KDDTest+"
)