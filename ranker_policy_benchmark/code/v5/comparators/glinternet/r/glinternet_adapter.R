options(warn = 2)

parse_args <- function(values) {
  result <- list()
  for (item in values) {
    if (!grepl("^--[a-z0-9-]+=", item)) stop(paste("Malformed argument:", item))
    key <- sub("^--([^=]+)=.*$", "\\1", item)
    value <- sub("^[^=]+=", "", item)
    if (!is.null(result[[key]])) stop(paste("Duplicate argument:", key))
    result[[key]] <- value
  }
  result
}

required_args <- c(
  "train-x", "train-y", "validation-x", "validation-y", "test-x",
  "candidate-pairs", "output-dir", "seed", "n-lambda", "lambda-min-ratio",
  "tolerance", "max-iter", "num-cores", "fixture-id"
)
args <- parse_args(commandArgs(trailingOnly = TRUE))
if (!setequal(names(args), required_args)) {
  stop(paste(
    "Argument set mismatch; missing=",
    paste(setdiff(required_args, names(args)), collapse = ","),
    "unknown=", paste(setdiff(names(args), required_args), collapse = ",")
  ))
}

read_x <- function(path) {
  frame <- read.csv(path, check.names = FALSE)
  matrix <- as.matrix(frame)
  storage.mode(matrix) <- "double"
  if (nrow(matrix) < 1 || ncol(matrix) < 2 || any(!is.finite(matrix))) {
    stop(paste("Invalid finite feature matrix:", path))
  }
  matrix
}

read_y <- function(path) {
  frame <- read.csv(path, check.names = FALSE)
  if (!identical(names(frame), "y")) stop(paste("Invalid response header:", path))
  values <- as.integer(frame$y)
  if (!all(values %in% c(0L, 1L)) || length(unique(values)) != 2L) {
    stop(paste("Response must contain both binary classes:", path))
  }
  values
}

X_train <- read_x(args[["train-x"]])
y_train <- read_y(args[["train-y"]])
X_validation <- read_x(args[["validation-x"]])
y_validation <- read_y(args[["validation-y"]])
X_test <- read_x(args[["test-x"]])
feature_names <- colnames(X_train)
if (!identical(colnames(X_validation), feature_names) ||
    !identical(colnames(X_test), feature_names)) {
  stop("Feature headers differ across train, validation, and test")
}
if (nrow(X_train) != length(y_train) ||
    nrow(X_validation) != length(y_validation)) {
  stop("Feature/response row-count mismatch")
}

pairs <- read.csv(args[["candidate-pairs"]], check.names = FALSE)
if (!identical(names(pairs), c("pair_id", "left_index", "right_index"))) {
  stop("Candidate-pair schema mismatch")
}
pairs$left_index <- as.integer(pairs$left_index)
pairs$right_index <- as.integer(pairs$right_index)
if (nrow(pairs) < 1 || any(pairs$left_index < 0L) ||
    any(pairs$right_index >= ncol(X_train)) ||
    any(pairs$left_index >= pairs$right_index) ||
    anyDuplicated(pairs$pair_id) ||
    anyDuplicated(paste(pairs$left_index, pairs$right_index, sep = ":"))) {
  stop("Candidate-pair identity or bounds violation")
}

seed <- as.integer(args[["seed"]])
n_lambda <- as.integer(args[["n-lambda"]])
lambda_min_ratio <- as.double(args[["lambda-min-ratio"]])
tolerance <- as.double(args[["tolerance"]])
max_iter <- as.integer(args[["max-iter"]])
num_cores <- as.integer(args[["num-cores"]])
if (!is.finite(seed) || n_lambda < 3L || !is.finite(lambda_min_ratio) ||
    lambda_min_ratio <= 0 || lambda_min_ratio >= 1 || !is.finite(tolerance) ||
    tolerance <= 0 || max_iter < 1L || num_cores != 1L) {
  stop("Invalid deterministic fit controls")
}

suppressPackageStartupMessages(library(glinternet))
if (as.character(packageVersion("glinternet")) != "1.0.13") {
  stop("Unexpected glinternet package version")
}
set.seed(seed)
fit <- glinternet(
  X_train,
  y_train,
  numLevels = rep(1L, ncol(X_train)),
  nLambda = n_lambda,
  lambdaMinRatio = lambda_min_ratio,
  interactionPairs = as.matrix(pairs[, c("left_index", "right_index")]) + 1L,
  family = "binomial",
  tol = tolerance,
  maxIter = max_iter,
  verbose = FALSE,
  numCores = num_cores
)

validation_probability <- predict(fit, X_validation, type = "response")
validation_probability <- matrix(
  validation_probability,
  nrow = nrow(X_validation),
  ncol = length(fit$lambda)
)
clipped <- pmin(pmax(validation_probability, 1e-15), 1 - 1e-15)
validation_loss <- colMeans(
  -(y_validation * log(clipped) + (1 - y_validation) * log(1 - clipped))
)
if (any(!is.finite(validation_loss))) stop("Non-finite validation loss")
minimum <- min(validation_loss)
chosen_index <- which(validation_loss <= minimum + 1e-12)[1]

effects <- coef(fit, lambdaIndex = chosen_index)[[1]]
scores <- rep(0, nrow(pairs))
active_pairs <- effects$interactions$contcont
active_coefs <- effects$interactionsCoef$contcont
if (!is.null(active_pairs)) {
  active_pairs <- matrix(active_pairs, ncol = 2)
  active_coefs <- as.double(unlist(active_coefs))
  if (nrow(active_pairs) != length(active_coefs)) {
    stop("Incomplete continuous-pair coefficient mapping")
  }
  for (index in seq_len(nrow(active_pairs))) {
    zero_based <- sort(as.integer(active_pairs[index, ]) - 1L)
    matches <- which(
      pairs$left_index == zero_based[1] & pairs$right_index == zero_based[2]
    )
    if (length(matches) != 1L) stop("Active pair is absent from declared universe")
    scores[matches] <- abs(active_coefs[index])
  }
}
if (any(!is.finite(scores))) stop("Non-finite interaction score")

test_probability <- as.double(
  predict(fit, X_test, type = "response", lambda = fit$lambda[chosen_index])
)
if (length(test_probability) != nrow(X_test) ||
    any(!is.finite(test_probability)) ||
    any(test_probability < 0) || any(test_probability > 1)) {
  stop("Invalid held-out prediction vector")
}

output_dir <- args[["output-dir"]]
if (dir.exists(output_dir) && length(list.files(output_dir, all.files = TRUE,
                                                no.. = TRUE)) > 0L) {
  stop("Refusing to overwrite non-empty output directory")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
validation_path <- data.frame(
  lambda_index = seq_along(fit$lambda) - 1L,
  lambda = as.double(fit$lambda),
  validation_log_loss = as.double(validation_loss),
  chosen = seq_along(fit$lambda) == chosen_index
)
pair_scores <- data.frame(
  pair_id = pairs$pair_id,
  left_index = pairs$left_index,
  right_index = pairs$right_index,
  score = scores,
  selected = scores > 0,
  score_direction = "higher_is_better"
)
predictions <- data.frame(
  row_id = seq_len(nrow(X_test)) - 1L,
  probability_class_1 = test_probability
)
adapter_meta <- data.frame(
  schema_version = 1L,
  adapter_id = "glinternet-r-file-v1",
  fixture_id = args[["fixture-id"]],
  package = "glinternet",
  package_version = as.character(packageVersion("glinternet")),
  package_source_sha256 = Sys.getenv("GLINTERNET_SOURCE_SHA256"),
  rocker_arm64_digest = Sys.getenv("ROCKER_ARM64_DIGEST"),
  family = "binomial",
  index_base = 0L,
  num_cores = num_cores,
  fit_path_calls = 1L,
  chosen_lambda_index = chosen_index - 1L,
  candidate_pair_count = nrow(pairs),
  test_row_count = nrow(X_test)
)
write.csv(validation_path, file.path(output_dir, "validation_path.csv"), row.names = FALSE)
write.csv(pair_scores, file.path(output_dir, "pair_scores.csv"), row.names = FALSE)
write.csv(predictions, file.path(output_dir, "predictions.csv"), row.names = FALSE)
write.csv(adapter_meta, file.path(output_dir, "adapter_meta.csv"), row.names = FALSE)
