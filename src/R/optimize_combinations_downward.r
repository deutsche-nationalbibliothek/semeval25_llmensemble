# File name: optimize_combinations_downward.r
# Description: Find optimal combinations from a set of combinations 
# by removing the one causing the least performance loss

library(tidyverse)
library(dplyr)
library(arrow)
library(aeneval)


# list files in results directory
# result_files <- list.files(
#   "results/", 
#   recursive = TRUE, 
#   pattern = ".*predictions.csv$",
#   full.names = TRUE)

# result_files <- setdiff(result_files, c("results//predictions.csv", "results//ranked_predictions.csv" ))

top60 <- read_csv("results/top60_combi.csv")

result_files <- paste0("results/", top60$model, "/", top60$prompt, "/predictions.csv")

names <- str_extract(result_files, pattern = "^results/+(.*)/predictions.csv$", group = 1)  |>
  str_replace("/", "_") 

result_files <- setNames(result_files, names)

#result_files <- result_files[1:10]

gold_standard <- read_csv(
  file = "datasets/all-subjects-tib-core-subjects-Article-Book-Conference-Report-Thesis-en-de-dev_sample1000.csv",
  col_select = c("idn", "dcterms:subject", "language", "text_type"),
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
    doc_id = idn,
    label_id = str_extract(
      label_id_messy,
      pattern = "(gnd:[0-9\\-X]+)"
    ),
    language = language,
    text_type = text_type
  ) |>
  distinct()


f1_at_5 <- function(predictions_set) {

  res = compute_set_retrieval_scores(
    gold_standard = gold_standard,
    predicted = filter(predictions_set, rank <= 5),
    mode = "micro"
  )
  
  value <- res |> 
    filter(metric == "f1") |>
    select(value)
  
  return(value)
}

pr_auc <- function(predictions_set) {
  
  res = compute_pr_auc(
    gold_standard = gold_standard,
    predicted = predictions_set,
    mode = "doc-avg"
  )
  
  res |> 
    select(value = pr_auc)
}

worst_model <- function(new_list_of_preds) {
  results <- new_list_of_preds |> 
    furrr::future_map_dfr(.f = pr_auc, .id = "model")
  
  a <- results |> 
    filter(value == max(value)) |> 
    slice_max(order_by = value, n = 1, with_ties = FALSE)
  
  message(paste("Score = ", round(a$value,3)))
  message(paste("Removing model = ", a$model))
  
  return(a)
  
}
compute_rank <- function(df) {
  df  |> 
    group_by(doc_id) |>
    mutate(rank = min_rank(desc(score))) |> 
    ungroup()
}

join_fun <- function(x, y, n_models) {
  full_join(x = x, y = y, by = c("doc_id", "label_id")) |>
    mutate_at(vars(starts_with("score")), ~replace_na(., 0)) |>
    transmute(
      doc_id, label_id,
      score = (score.x + n_models * score.y) / (n_models + 1))
}


# take a fused list of predictions and substract the predictions of a single model
# from the joined score and anti_join the single model predictions
anti_join_fun <- function(joined_preds, single_model_preds, n_models, .verbose = FALSE) {

  # common_preds <- inner_join(
  #   joined_preds, 
  #   single_model_preds, 
  #   c("doc_id", "label_id"))  |>
  #   select(-contains("score"), -contains("rank")) 
  
  # singel_model_only_preds <- single_model_preds  |>
  #   anti_join(y = joined_preds, c("doc_id", "label_id"))
  
  eps <- 1e-10

  res <- joined_preds  |>
    left_join(single_model_preds, by = c("doc_id", "label_id")) |>
    mutate(score.y = replace_na(score.y, 0)) |>
    mutate(
      doc_id, label_id,
      score = pmin(pmax((n_models * score.x - score.y) / (n_models - 1), 0), 1)) 

    n_preds_to_remove <- sum(res[["score"]] < eps)  
    if (.verbose)
      print(paste("Removed predictions: ", n_preds_to_remove))
    # when score is (almost zero) score.y was the major contributor, so we remove it
    res <- res  |> filter(abs(score) > eps)  |>
    compute_rank() 
    # filter(score > 1)


  return(select(res, doc_id, label_id, score))
}


library(future)

plan(multicore, workers = 40)
modell_kette <- names(result_files)
exclusion_chain <- character(0)
score_kette <- numeric(0)
list_of_predictions <- result_files  |>
  map(.f = arrow::read_feather) |> 
  map(.f = compute_rank)

joined_pred_set <- list_of_predictions  |>
  map_dfr(.f = ~.x) |>
  group_by(doc_id, label_id) |>
  summarise(score = sum(score))  |>
  ungroup() |>
  mutate(score = score/length(list_of_predictions))  |>
  compute_rank()

initial_pr_auc <- pr_auc(joined_pred_set)

message("Initial PR-AUC: ", initial_pr_auc)

# anti_join_fun(
#   single_model_preds = list_of_predictions[["mistral-7B-0p3_alltypes-de-abstitle-8-4"]],
#   joined_preds = joined_pred_set2, 
#   n_models = length(modell_kette))  |>
#   View()

for (i in 1:50) {
  # keep all items in list_of_predictions that are not in modell_kette
  list_of_predictions <- list_of_predictions  |>
    keep(names(list_of_predictions) %in% modell_kette)

  print(paste("lenght modell_kette", length(modell_kette)))
  print(paste("lenght list_of_predictions", length(list_of_predictions)))
  
  pred_set_candidates <- list_of_predictions  |>
    furrr::future_map(.f = ~anti_join_fun(joined_preds = joined_pred_set, single_model_preds = .x, n_models = length(modell_kette)))

  bst_mdl <- worst_model(pred_set_candidates)
  neuer_streichkandidat <- bst_mdl$model
  neuer_score <- bst_mdl$value
  score_kette <- c(score_kette, neuer_score)
  # update list_of_predictions by joining the current best to the list
  joined_pred_set <- joined_pred_set  |>
    anti_join_fun(
      single_model_preds = list_of_predictions[[neuer_streichkandidat]],
      n_models = length(modell_kette),
      .verbose = TRUE
    )

  modell_kette <- setdiff(modell_kette, neuer_streichkandidat)
  exclusion_chain <- c(exclusion_chain, neuer_streichkandidat)
}

data.frame(
  model_prompt = exclusion_chain,
  score = score_kette
  ) |> 
  write_csv("results/exclusion_chain.csv")
