#!/usr/bin/env Rscript
# File name: compute_metrics.r
# Description: Computes metrics for results of individual models and ensembles

library("optparse")

option_list <- list(
  make_option(
    c("--ground_truth"),
    type = "character",
    default = "datasets/all-subjects-tib-core-subjects-Article-Book-Conference-Report-Thesis-en-de-dev_sample1000.csv",
    help = "path to the ground truth file",
    metavar = "character"
  ),
  make_option(
    c("--predictions"),
    type = "character", default = "results/predictions.arrow",
    help = "path to the predictions file",
    metavar = "character"
  ),
  make_option(
    c("--ordinal_ranking"),
    type = "logical", default = FALSE,
    help = "pr-curves is calculated using ordinal ranking (limits) 
      without thresholds on confidence scores",
  ),
  make_option(
    c("--set_retrieval_only"),
    type = "logical", default = FALSE,
    help = "Produce no ranked retrieval metrics, only set retrieval metrics"
  ),
  make_option(
    c("--ignore_unmatched_docs"),
    type = "logical", default = FALSE,
    help = "If set to true, documents in the ground truth that are 
      not in the predictions and vice versa are ignored.",
  ),
  make_option(
    c("--out_folder"),
    type = "character", default = "results/",
    help = "path to the output folder",
    metavar = "character"
  ),
  make_option(
    c("--n_jobs"), type = "integer", default = 20,
    help = "number of jobs to run in parallel",
    metavar = "integer"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(future))
suppressPackageStartupMessages(library(arrow))
library(aeneval)

# Lese Vorschläge von Attention-XML Test
message("loading predictions...")
predicted <- read_csv(opt$predictions, show_col_types = FALSE)
if (!opt$set_retrieval_only) {
  predicted <- predicted  |>
    group_by(doc_id)  |>
    mutate(rank = min_rank(-score))  |>
    ungroup()
}

# Lese Gold-Standard für gesamten Korpus
message("loading ground truth...")
gold_standard <- read_csv(
  file = opt$ground_truth,
  col_select = c("doc_id", "dcterms:subject", "language", "text_type"),
  col_types = "cccc"
) |>
  separate_wider_delim(
    cols = `dcterms:subject`,
    names_sep = "_",
    delim = ",",
    too_few = "align_start"
  ) |>
  pivot_longer(
    cols = starts_with("dcterms:subject"),
    names_to = "foo",
    values_to = "label_id_messy",
    values_drop_na = TRUE
  ) |>
  transmute(
    doc_id,
    label_id = str_extract(
      label_id_messy,
      pattern = "(gnd:[0-9\\-X]+)"
    ),
    language = language,
    text_type = text_type
  ) |>
  distinct()

non_unique_tuples <- gold_standard  |>
  group_by(doc_id, label_id)  |>
  filter(n() > 1)  |>
  nrow()

if (non_unique_tuples > 0)
  stop("There are non-unique tuples (doc_id, label_id) in the ground truth.") 

missing_labels <- gold_standard  |>
  filter(is.na(label_id))  |>
  nrow()

stopifnot(missing_labels == 0)

n_not_indexed <- anti_join(
  x = distinct(gold_standard, doc_id),
  y = distinct(predicted, doc_id),
  by = "doc_id"
)  |> nrow()

if (n_not_indexed > 0) {
  message(
    "There are ",
    n_not_indexed,
    " documents in the ground truth with no predictions."
  )
}

if (opt$ignore_unmatched_docs) {
  message("Ignoring unmatched documents.")
  gold_standard <- gold_standard  |>
    semi_join(
      y = distinct(predicted, doc_id),
      by = "doc_id"
    )
  predicted <- predicted  |>
    semi_join(
      y = distinct(gold_standard, doc_id),
      by = "doc_id"
    )

  if (nrow(gold_standard) == 0) {
    stop("No documents in ground truth that match predictions.")
  }
  if (nrow(predicted) == 0) {
    stop("No documents in predictions that match ground truth.")
  }
}

if (!opt$set_retrieval_only) {
  max_preds <- predicted  |>
    group_by(doc_id)  |>
    summarise(n_preds = n())  |>
    summarise(max_preds = max(n_preds))  |>
    pull(max_preds)

  message("Switching to multicore exectution with ", opt$n_jobs, " workers.")
  plan(multicore, workers = opt$n_jobs)

  # Berechne die Precision-Recall-Kurve
  message("compute pr-curve...")
  if (opt$ordinal_ranking) {
    pr_curve <- compute_pr_curve(
      gold_standard,
      predicted,
      limit_range = c(1:max_preds),
      steps = 1,
      optimize_cutoff = TRUE,
      .verbose = TRUE,
      .progress = TRUE
    )
  } else {
    pr_curve <- compute_pr_curve(
      gold_standard,
      predicted,
      steps = 100,
      limit_range = c(1:max_preds),
      optimize_cutoff = TRUE,
      .verbose = TRUE,
      .progress = TRUE
    )
  }


  write_csv(select(pr_curve$plot_data,
                  recall = rec,
                  precision = prec_cummax),
            file.path(opt$out_folder, "pr_curve.csv"))

  # Berechne den Precision Recall AUC mit Konfidenzintervallen
  message("compute pr-auc...")
  pr_auc <- compute_pr_auc_from_curve(
    pr_curve
  )

  json_output_pr_auc <- pr_auc |>
    transmute(pr_auc) |>
    rjson::toJSON() |>
    write_lines(file = file.path(opt$out_folder, "pr_auc.json"))

  message("generating plot")
  g <- ggplot(pr_curve$plot_data, aes(x = rec, y = prec_cummax)) +
    geom_point() +
    geom_line() +
    ggtitle(paste0("Precision-Recall-Kurve", opt$opt$evalset, "-Set"),
            paste("prAUC =", round(pr_auc$pr_auc[1], 3))) +
    coord_fixed(xlim = c(0,1)) +
    xlab("Recall") +
    ylab("Precision")

  ggsave(filename =  file.path(opt$out_folder, "pr_curve_plot.svg"),
        g, device = "svg")

  message("Limiting predictions to opt threshols and limit:")
  threshold <- pull(pr_curve$opt_cutoff, thresholds)
  message("  Threshold = ", threshold)
  limit <- pull(pr_curve$opt_cutoff, limits)
  message("  Limit = ", limit)
  predicted <- filter(predicted, rank <= limit & score >= threshold)
}



# Berechne die Retrieval Metriken 
# mit Konfidenzintervallen
message("compute set retrieval scores...")
res_overall <- aeneval::compute_set_retrieval_scores(
  gold_standard,
  predicted
)  |> mutate(
  language = "all",
  text_type = "all"
)

res_by_language <- aeneval::compute_set_retrieval_scores(
  gold_standard,
  predicted,
  doc_strata = "language"
)  |> mutate(
  text_type = "all"
)

res_by_texttype <- aeneval::compute_set_retrieval_scores(
  gold_standard,
  predicted,
  doc_strata = "text_type"
)  |> mutate(
  language = "all"
)

res_by_texttype_and_language <- aeneval::compute_set_retrieval_scores(
  gold_standard,
  predicted,
  doc_strata = c("text_type", "language")
)

# bringe die Ergebnisse in ein für "dvc metrics" günstiges Format
print_metric_to_json <- function(metric_name) {
  if (!opt$set_retrieval_only) {
    file_suffix <- "_opt.json"
    pivoting_template <- "{metric}_{.value}"
  } else {
    file_suffix <- ".json"
    pivoting_template <- "{metric}_{.value}"
  }

  res_overall |>
    transmute(metric, value) |>
    filter(metric == metric_name) |>
    pivot_wider(
      names_from = metric,
      values_from = c(value),
      names_glue = pivoting_template
    ) |>
    rjson::toJSON() |>
    write_lines(
      file = file.path(opt$out_folder, paste0(metric_name, file_suffix))
    )
}

json_output <- res_overall$metric |>
  purrr::map(.f = print_metric_to_json)

stratified_results <- rbind(
  res_overall,
  res_by_language,
  res_by_texttype,
  res_by_texttype_and_language
)

write_feather(
  stratified_results,
  file.path(opt$out_folder, "stratified_results.arrow")
)

# pairwise_comparison <- create_comparison(gold_standard, predictions_at_5)
# write_feather(pairwise_comparison, sink = file.path("intermediate-results", opt$opt$evalset, "pairwise_comparison.arrow"))
# results_doc_wise <- compute_intermediate_results(pairwise_comparison, grouping_var = "doc_id")
# write_feather(results_doc_wise$results_table, sink = file.path("intermediate-results", opt$opt$evalset, "results_doc_wise.arrow"))
