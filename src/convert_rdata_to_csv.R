#!/usr/bin/env Rscript

# MARLIN RData to CSV Converter
# Converts MARLIN RData files (betas.RData, features) to CSV format for PyTorch training

library(data.table)

# Function to convert betas.RData and y.RData to CSV
# This follows the exact preprocessing from MARLIN_training.R
convert_marlin_to_csv <- function(betas_file, y_file, output_csv, merge_pb_controls = TRUE) {
  cat("Loading MARLIN data files...\n")
  cat("  Betas:", betas_file, "\n")
  cat("  Labels:", y_file, "\n")

  # Load betas matrix
  load(betas_file)
  if (!exists("betas")) {
    stop("Could not find 'betas' object in ", betas_file)
  }

  # Load class labels
  load(y_file)
  if (!exists("y")) {
    stop("Could not find 'y' object in ", y_file)
  }

  cat("Data dimensions:", nrow(betas), "samples ×", ncol(betas), "features\n")
  cat("Number of classes:", length(unique(y)), "\n")
  cat("Class distribution:\n")
  print(table(y))

  # Apply MARLIN preprocessing: merge peripheral blood controls
  if (merge_pb_controls) {
    y <- as.character(y)
    original_classes <- length(unique(y))
    y <- ifelse(grepl("^PB", y) == TRUE, "PB controls", y)
    y <- as.factor(y)
    cat("\nMerged PB controls:", original_classes, "→", length(unique(y)), "classes\n")
    cat("Class distribution after merging:\n")
    print(table(y))
  }

  # Convert to data.table
  dt <- as.data.table(betas)

  # Add label column at the beginning (matching MARLIN training format)
  dt[, label := as.character(y)]
  setcolorder(dt, c("label"))

  # Write to CSV
  cat("\nWriting to CSV:", output_csv, "\n")
  fwrite(dt, output_csv, row.names = FALSE)

  cat("Conversion complete!\n")
  cat("Output file:", output_csv, "\n")
  cat("Dimensions:", nrow(dt), "rows ×", ncol(dt), "columns\n")
  cat("  Label column + ", ncol(dt) - 1, " CpG features\n")

  return(dt)
}

# Function to convert betas.RData to CSV (legacy, for single file conversion)
convert_betas_to_csv <- function(rdata_file, output_csv, class_labels = NULL) {
  cat("Loading RData file:", rdata_file, "\n")

  # Load RData file
  load(rdata_file)

  # Find the main data matrix (usually named 'betas' or similar)
  data_objects <- ls()
  cat("Objects in RData:", paste(data_objects, collapse = ", "), "\n")

  # Try to find the beta matrix
  if (exists("betas")) {
    betas <- get("betas")
  } else if (exists("data")) {
    betas <- get("data")
  } else if (exists("beta_values")) {
    betas <- get("beta_values")
  } else {
    stop("Could not find beta matrix. Available objects: ", paste(data_objects, collapse = ", "))
  }

  cat("Data dimensions:", nrow(betas), "samples ×", ncol(betas), "features\n")

  # Convert to data.table for efficient processing
  dt <- as.data.table(betas)

  # Add class labels if provided
  if (!is.null(class_labels)) {
    if (length(class_labels) == nrow(betas)) {
      dt[, label := class_labels]
      setcolorder(dt, c("label"))
    } else {
      warning("Length of class_labels does not match number of samples. Labels not added.")
    }
  }

  # Write to CSV
  cat("Writing to CSV:", output_csv, "\n")
  fwrite(dt, output_csv, row.names = FALSE)

  cat("Conversion complete!\n")
  cat("Output file:", output_csv, "\n")
  cat("Dimensions:", nrow(dt), "rows ×", ncol(dt), "columns\n")

  return(dt)
}

# Function to extract feature names from RData
extract_features <- function(rdata_file, output_file) {
  cat("Loading RData file:", rdata_file, "\n")

  load(rdata_file)

  # List available objects for debugging
  data_objects <- ls()
  cat("Available objects:", paste(data_objects, collapse = ", "), "\n")

  # Try to find feature names (check common names)
  features <- NULL

  if (exists("betas_sub_names")) {
    features <- get("betas_sub_names")
    cat("Found 'betas_sub_names' object\n")
  } else if (exists("features")) {
    features <- get("features")
    cat("Found 'features' object\n")
  } else if (exists("feature_names")) {
    features <- get("feature_names")
    cat("Found 'feature_names' object\n")
  } else if (exists("probes")) {
    features <- get("probes")
    cat("Found 'probes' object\n")
  } else if (exists("cpg_sites")) {
    features <- get("cpg_sites")
    cat("Found 'cpg_sites' object\n")
  } else if (exists("probe_names")) {
    features <- get("probe_names")
    cat("Found 'probe_names' object\n")
  } else {
    # Try to get column names from any matrix object
    cat("Checking for matrix/data.frame objects with column names...\n")
    for (obj_name in data_objects) {
      obj <- get(obj_name)
      if (is.matrix(obj) || is.data.frame(obj)) {
        if (!is.null(colnames(obj))) {
          features <- colnames(obj)
          cat("Found column names from object:", obj_name, "\n")
          break
        }
      }
    }
  }

  if (is.null(features)) {
    stop("Could not find feature names in RData file.\nAvailable objects: ",
         paste(data_objects, collapse = ", "))
  }

  cat("Found", length(features), "features\n")

  # Show first few features as example
  cat("First 5 features:", paste(head(features, 5), collapse = ", "), "\n")

  # Write to file (one feature per line)
  writeLines(as.character(features), output_file)

  cat("Features saved to:", output_file, "\n")

  return(features)
}

# Function to convert y.RData to labels CSV
convert_y_to_csv <- function(y_file, output_csv, betas_file = NULL, merge_pb_controls = TRUE) {
  cat("Loading y.RData file:", y_file, "\n")

  # Load class labels
  load(y_file)
  if (!exists("y")) {
    stop("Could not find 'y' object in ", y_file)
  }

  # Get sample IDs from betas if provided
  sample_ids <- NULL
  if (!is.null(betas_file)) {
    cat("Loading betas.RData to get sample IDs:", betas_file, "\n")
    load(betas_file)
    if (exists("betas") && !is.null(rownames(betas))) {
      sample_ids <- rownames(betas)
    }
  }

  # If no sample IDs from betas, create sequential IDs
  if (is.null(sample_ids)) {
    sample_ids <- paste0("sample_", seq_len(length(y)))
  }

  cat("Number of samples:", length(y), "\n")
  cat("Original class distribution:\n")
  print(table(y))

  # Create data table with original labels
  dt <- data.table(
    sample_id = sample_ids,
    original_label = as.character(y)
  )

  # Add merged labels if requested
  if (merge_pb_controls) {
    merged_labels <- as.character(y)
    merged_labels <- ifelse(grepl("^PB", merged_labels), "PB controls", merged_labels)

    dt[, merged_label := merged_labels]

    cat("\nAfter merging PB controls:\n")
    print(table(merged_labels))

    # Add a flag for which samples were merged
    dt[, pb_merged := grepl("^PB", original_label)]
  }

  # Add numeric class IDs for original labels
  unique_labels <- unique(dt$original_label)
  label_mapping <- data.table(
    original_label = unique_labels,
    class_id = seq_len(length(unique_labels)) - 1  # 0-based indexing
  )
  dt <- merge(dt, label_mapping, by = "original_label", all.x = TRUE)

  # Reorder columns
  if (merge_pb_controls) {
    setcolorder(dt, c("sample_id", "original_label", "merged_label", "class_id", "pb_merged"))
  } else {
    setcolorder(dt, c("sample_id", "original_label", "class_id"))
  }

  # Write to CSV
  cat("\nWriting to CSV:", output_csv, "\n")
  fwrite(dt, output_csv, row.names = FALSE)

  cat("Labels CSV created successfully!\n")
  cat("Columns:", paste(names(dt), collapse = ", "), "\n")
  cat("Total samples:", nrow(dt), "\n")

  # Print summary
  if (merge_pb_controls) {
    cat("\nSummary:\n")
    cat("  Original classes:", length(unique(dt$original_label)), "\n")
    cat("  Merged classes:", length(unique(dt$merged_label)), "\n")
    cat("  PB controls merged:", sum(dt$pb_merged), "samples\n")
  }

  return(dt)
}

# Function to create sample labels CSV from metadata (legacy)
create_labels_csv <- function(sample_ids, labels, output_csv) {
  dt <- data.table(
    sample_id = sample_ids,
    label = labels
  )

  fwrite(dt, output_csv)
  cat("Sample labels saved to:", output_csv, "\n")

  return(dt)
}

# Main execution
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  if (length(args) == 0) {
    cat("
MARLIN RData to CSV Converter

Usage:
  Rscript convert_rdata_to_csv.R <mode> <inputs...> <output> [options]

Modes:
  marlin_to_csv    Convert MARLIN betas.RData + y.RData to training CSV (RECOMMENDED)
  y_to_csv         Convert y.RData to labels CSV with original info (NEW)
  betas_to_csv     Convert beta values matrix to CSV (without labels)
  extract_features Extract feature names to text file
  create_labels    Create sample labels CSV

Examples:
  # Convert original MARLIN training data (RECOMMENDED):
  Rscript convert_rdata_to_csv.R marlin_to_csv betas.RData y.RData training_data.csv

  # Convert y.RData to labels CSV (with original and merged labels):
  Rscript convert_rdata_to_csv.R y_to_csv y.RData labels.csv

  # Convert y.RData with sample IDs from betas:
  Rscript convert_rdata_to_csv.R y_to_csv y.RData labels.csv betas.RData

  # With full paths:
  Rscript convert_rdata_to_csv.R marlin_to_csv \\
      ../MARLIN/betas.RData \\
      ../MARLIN/y.RData \\
      training_data.csv

  # Extract feature names
  Rscript convert_rdata_to_csv.R extract_features marlin_v1.features.RData reference_features.txt

  # Convert only betas (if you have labels separately)
  Rscript convert_rdata_to_csv.R betas_to_csv betas.RData data_only.csv

Arguments:
  mode    - Conversion mode
  inputs  - Input RData file(s)
  output  - Output CSV/text file

Note: For MARLIN training, use 'marlin_to_csv' mode which properly handles
      both betas.RData and y.RData, including the PB controls merging.

")
    quit(save = "no", status = 0)
  }

  mode <- args[1]

  if (mode == "marlin_to_csv") {
    if (length(args) < 4) {
      stop("Usage: Rscript convert_rdata_to_csv.R marlin_to_csv <betas.RData> <y.RData> <output.csv>")
    }

    betas_file <- args[2]
    y_file <- args[3]
    output_file <- args[4]

    convert_marlin_to_csv(betas_file, y_file, output_file)

  } else if (mode == "y_to_csv") {
    if (length(args) < 3) {
      stop("Usage: Rscript convert_rdata_to_csv.R y_to_csv <y.RData> <output.csv> [betas.RData]")
    }

    y_file <- args[2]
    output_file <- args[3]
    betas_file <- if (length(args) >= 4) args[4] else NULL

    convert_y_to_csv(y_file, output_file, betas_file = betas_file)

  } else if (mode == "betas_to_csv") {
    if (length(args) < 3) {
      stop("Usage: Rscript convert_rdata_to_csv.R betas_to_csv <input.RData> <output.csv>")
    }

    input_file <- args[2]
    output_file <- args[3]

    convert_betas_to_csv(input_file, output_file)

  } else if (mode == "extract_features") {
    if (length(args) < 3) {
      stop("Usage: Rscript convert_rdata_to_csv.R extract_features <input.RData> <output.txt>")
    }

    input_file <- args[2]
    output_file <- args[3]

    extract_features(input_file, output_file)

  } else if (mode == "create_labels") {
    if (length(args) < 3) {
      stop("Usage: Rscript convert_rdata_to_csv.R create_labels <input.RData> <output.csv>")
    }

    input_file <- args[2]
    output_file <- args[3]

    # Load RData and try to find sample IDs and labels
    load(input_file)

    # This is specific to MARLIN data structure
    # Adjust based on actual data structure
    if (exists("sample_ids") && exists("labels")) {
      create_labels_csv(sample_ids, labels, output_file)
    } else {
      cat("Could not find sample_ids and labels in RData file\n")
      cat("Available objects:", paste(ls(), collapse = ", "), "\n")
    }

  } else {
    stop("Unknown mode: ", mode)
  }
}

# Run main function
if (!interactive()) {
  main()
}
