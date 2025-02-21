library(tidyverse)
library(arrow)
library(aeneval)


# list files in results directory
result_files <- list.files(
  "results/", 
  recursive = TRUE, 
  pattern = ".*predictions.csv$",
  full.names = TRUE)

result_files <- setdiff(result_files, c("results//predictions.arrow", "results//ranked_predictions.arrow" ))

names <- str_extract(result_files, pattern = "^results//(.*)/predictions.arrow$", group = 1)  |>
  str_replace("/", "_") 

result_files <- setNames(result_files, names)

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

best_model <- function(new_list_of_preds) {
  results <- new_list_of_preds |> 
    furrr::future_map_dfr(.f = pr_auc, .id = "model")
  
  a <- results |> 
    filter(value == max(value)) |> 
    slice_max(order_by = value, n = 1, with_ties = FALSE)
  
  message(paste("Score = ", round(a$value,3)))
  message(paste("Adding model = ", a$model))
  
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
    transmute(doc_id, label_id, score = (score.x + n_models*score.y)/(n_models + 1)) |> 
    compute_rank()
}

library(future)

plan(multicore, workers = 40)
modell_kette <- character(0)
score_kette <- numeric(0)
list_of_predictions <- result_files  |>
  map(.f = arrow::read_feather) |> 
  map(.f = compute_rank)

for (i in 1:70) {
  # keep all items in list_of_predictions that are not in modell_kette
  list_of_predictions <- list_of_predictions  |>
    discard(names(list_of_predictions) %in% modell_kette)
  
  bst_mdl <- best_model(list_of_predictions)
  neuer_bester <- bst_mdl$model
  neuer_score <- bst_mdl$value
  modell_kette <- c(modell_kette, neuer_bester)
  score_kette <- c(score_kette, neuer_score)
  # update list_of_predictions by joining the current best to the list
  list_of_predictions <- list_of_predictions  |>
    furrr::future_map(.f = ~join_fun(x = .x, y = list_of_predictions[[neuer_bester]], n_models = length(modell_kette)))

}

data.frame(
  model_prompt = modell_kette,
  cum_score = score_kette
  ) |> 
  write_csv("results/model_chain.csv")
