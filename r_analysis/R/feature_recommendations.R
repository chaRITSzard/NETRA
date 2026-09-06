# NETRA — Phase 1 feature recommendation handoff

options(stringsAsFactors = FALSE)

REPORT_DIR <- "r_analysis/reports"
dir.create(
  REPORT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

# Locked feature recommendations
features <- c(
  "duration",
  "protocol_type",
  "service",
  "flag",
  "src_bytes",
  "dst_bytes",
  "land",
  "wrong_fragment",
  "urgent",
  "hot",
  "num_failed_logins",
  "logged_in",
  "num_compromised",
  "root_shell",
  "su_attempted",
  "num_root",
  "num_file_creations",
  "num_shells",
  "num_access_files",
  "num_outbound_cmds",
  "is_host_login",
  "is_guest_login",
  "count",
  "srv_count",
  "serror_rate",
  "srv_serror_rate",
  "rerror_rate",
  "srv_rerror_rate",
  "same_srv_rate",
  "diff_srv_rate",
  "srv_diff_host_rate",
  "dst_host_count",
  "dst_host_srv_count",
  "dst_host_same_srv_rate",
  "dst_host_diff_srv_rate",
  "dst_host_same_src_port_rate",
  "dst_host_srv_diff_host_rate",
  "dst_host_serror_rate",
  "dst_host_srv_serror_rate",
  "dst_host_rerror_rate",
  "dst_host_srv_rerror_rate"
)

# Derive feature types from the actual feature names.
categorical_features <- c(
  "protocol_type",
  "service",
  "flag"
)

feature_type <- ifelse(
  features %in% categorical_features,
  "categorical",
  "numeric"
)

# Decisions
decision <- c(
  "RE-ENCODE",  # duration
  "KEEP",       # protocol_type
  "RE-ENCODE",  # service
  "KEEP",       # flag
  "RE-ENCODE",  # src_bytes
  "RE-ENCODE",  # dst_bytes
  "KEEP",       # land
  "KEEP",       # wrong_fragment
  "KEEP",       # urgent
  "KEEP",       # hot
  "KEEP",       # num_failed_logins
  "KEEP",       # logged_in
  "DROP",       # num_compromised
  "KEEP",       # root_shell
  "KEEP",       # su_attempted
  "KEEP",       # num_root
  "KEEP",       # num_file_creations
  "KEEP",       # num_shells
  "KEEP",       # num_access_files
  "DROP",       # num_outbound_cmds
  "KEEP",       # is_host_login
  "KEEP",       # is_guest_login
  "KEEP",       # count
  "KEEP",       # srv_count
  "KEEP",       # serror_rate
  "DROP",       # srv_serror_rate
  "KEEP",       # rerror_rate
  "DROP",       # srv_rerror_rate
  "KEEP",       # same_srv_rate
  "KEEP",       # diff_srv_rate
  "KEEP",       # srv_diff_host_rate
  "KEEP",       # dst_host_count
  "DROP",       # dst_host_srv_count
  "KEEP",       # dst_host_same_srv_rate
  "KEEP",       # dst_host_diff_srv_rate
  "KEEP",       # dst_host_same_src_port_rate
  "KEEP",       # dst_host_srv_diff_host_rate
  "DROP",       # dst_host_serror_rate
  "DROP",       # dst_host_srv_serror_rate
  "DROP",       # dst_host_rerror_rate
  "DROP"        # dst_host_srv_rerror_rate
)


# Rationale
rationale <- c(
  "Heavy-tailed raw distribution; use log1p.",
  "Cramér's V = 0.3186272; 3 categories; one-hot.",
  "Cramér's V = 0.5578667; 70 training categories; frequency encode initially.",
  "Cramér's V = 0.4898992; 11 categories; one-hot.",
  "Extremely heavy-tailed raw distribution; use log1p.",
  "Extremely heavy-tailed raw distribution; use log1p.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain rare signal.",
  "Significant and variable; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Redundant with num_root; r = 0.9988335.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Representative of num_compromised/num_root pair.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Zero variance; drop.",
  "BH-adjusted p approximately 0.9287; retain for model-based validation.",
  "r = 0.8602881 with hot, but distinct semantics; retain.",
  "Strong class-dependent distribution; retain.",
  "Strong class-dependent distribution; retain.",
  "Representative of serror-rate family.",
  "Redundant serror-rate family; r = 0.9932892 with serror_rate.",
  "Representative of rerror-rate family.",
  "Redundant rerror-rate family; r = 0.9890077 with rerror_rate.",
  "Strong class-dependent distribution; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Redundant with dst_host_same_srv_rate; r = 0.8966635.",
  "Representative of dst_host service-count/rate pair.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Significant class differences; retain.",
  "Redundant serror-rate family.",
  "Redundant serror-rate family.",
  "Redundant rerror-rate family.",
  "Redundant rerror-rate family."
)


# Validation
stopifnot(length(features) == 41)
stopifnot(length(feature_type) == 41)
stopifnot(length(decision) == 41)
stopifnot(length(rationale) == 41)

stopifnot(!anyDuplicated(features))

stopifnot(
  all(
    decision %in% c(
      "KEEP",
      "RE-ENCODE",
      "DROP"
    )
  )
)


# Build final table
recommendations <- data.frame(
  feature = features,
  feature_type = feature_type,
  decision = decision,
  rationale = rationale
)

# Write CSV
write.csv(
  recommendations,
  file.path(
    REPORT_DIR,
    "feature_recommendations.csv"
  ),
  row.names = FALSE
)

cat("\n==============================\n")
cat("FEATURE RECOMMENDATIONS WRITTEN\n")
cat("==============================\n\n")

cat(
  "Features:",
  nrow(recommendations),
  "\n\n"
)

print(
  table(
    recommendations$decision
  )
)

cat("\n")