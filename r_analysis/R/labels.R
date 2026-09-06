# NSL-KDD attack class mapping

source("r_analysis/R/load_data.R")

attack_class_map <- c(
  # Normal
  "normal" = "normal",

  # DoS
  "back" = "DoS",
  "land" = "DoS",
  "neptune" = "DoS",
  "pod" = "DoS",
  "smurf" = "DoS",
  "teardrop" = "DoS",
  "mailbomb" = "DoS",
  "processtable" = "DoS",
  "udpstorm" = "DoS",
  "apache2" = "DoS",
  "worm" = "DoS",

  # Probe
  "satan" = "Probe",
  "ipsweep" = "Probe",
  "nmap" = "Probe",
  "portsweep" = "Probe",
  "mscan" = "Probe",
  "saint" = "Probe",

  # R2L
  "guess_passwd" = "R2L",
  "ftp_write" = "R2L",
  "imap" = "R2L",
  "phf" = "R2L",
  "multihop" = "R2L",
  "warezmaster" = "R2L",
  "warezclient" = "R2L",
  "spy" = "R2L",
  "named" = "R2L",
  "sendmail" = "R2L",
  "snmpgetattack" = "R2L",
  "snmpguess" = "R2L",
  "xlock" = "R2L",
  "xsnoop" = "R2L",
  "httptunnel" = "R2L",
  "sqlattack" = "R2L",
  "ps" = "R2L",

  # U2R
  "buffer_overflow" = "U2R",
  "loadmodule" = "U2R",
  "rootkit" = "U2R",
  "perl" = "U2R",
  "xterm" = "U2R"
)

assign_attack_class <- function(data) {

  raw_labels <- as.character(data$label)

  unmapped <- setdiff(unique(raw_labels), names(attack_class_map))

  if (length(unmapped) > 0) {
    stop(
      "Unmapped attack labels: ",
      paste(unmapped, collapse = ", ")
    )
  }

  data$attack_class <- unname(
    attack_class_map[raw_labels]
  )

  data$attack_class <- factor(
    data$attack_class,
    levels = c("normal", "DoS", "Probe", "R2L", "U2R")
  )

  data
}

train_data <- assign_attack_class(train_data)
test_data <- assign_attack_class(test_data)

cat("\nTraining 5-class distribution:\n")
print(table(train_data$attack_class))

cat("\nTest 5-class distribution:\n")
print(table(test_data$attack_class))

cat("\nMapped labels:\n")
print(attack_class_map)