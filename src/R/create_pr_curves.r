library(arrow)
library(tidyverse)

df <- read_feather("results/stratified_results_by_model_and_prompt.arrow")

df  |>
  filter(metric == "f1", language == "all", text_type == "all")  |>
  arrange(desc(value))  |>
  head()

file_paths_curves <- c(
  "results/top20-models-and-prompts/pr_curve.csv",
  "results/one-model-all-prompts/pr_curve.csv",
  "results/one-prompt-all-models/pr_curve.csv",
  "results/one-prompt-one-model/pr_curve.csv"
)

file_paths_recall <- c(
  "results/top20-models-and-prompts/rec_opt.json",
  "results/one-model-all-prompts/rec_opt.json",
  "results/one-prompt-all-models/rec_opt.json",
  "results/one-prompt-one-model/rec_opt.json"
)

file_paths_precison <- c(
  "results/top20-models-and-prompts/prec_opt.json",
  "results/one-model-all-prompts/prec_opt.json",
  "results/one-prompt-all-models/prec_opt.json",
  "results/one-prompt-one-model/prec_opt.json"
)

file_paths_f1 <- c(
  "results/top20-models-and-prompts/f1_opt.json",
  "results/one-model-all-prompts/f1_opt.json",
  "results/one-prompt-all-models/f1_opt.json",
  "results/one-prompt-one-model/f1_opt.json"
)

file_paths_pr_auc <- c(
  "results/top20-models-and-prompts/pr_auc.json",
  "results/one-model-all-prompts/pr_auc.json",
  "results/one-prompt-all-models/pr_auc.json",
  "results/one-prompt-one-model/pr_auc.json"
)


file_paths_curves <- setNames(file_paths_curves, c("top-20-ensemble", "one-model-all-prompts", "one-prompt-all-models", "one-model-one-prompt"))
file_paths_precison <- setNames(file_paths_precison, c("top-20-ensemble", "one-model-all-prompts", "one-prompt-all-models", "one-model-one-prompt"))
file_paths_recall <- setNames(file_paths_recall, c("top-20-ensemble", "one-model-all-prompts", "one-prompt-all-models", "one-model-one-prompt"))
file_paths_f1 <- setNames(file_paths_f1, c("top-20-ensemble", "one-model-all-prompts", "one-prompt-all-models", "one-model-one-prompt"))
file_paths_pr_auc <- setNames(file_paths_pr_auc, c("top-20-ensemble", "one-model-all-prompts", "one-prompt-all-models", "one-model-one-prompt"))

pr_curve_data <- file_paths_curves  |>
  map(read_csv, show_col_types = FALSE)  |>
  bind_rows(.id = "ensemble_type")

pr_auc <- file_paths_pr_auc  |>
  map_dfr(
    .f = ~ data.frame(
      pr_auc = jsonlite::read_json(.x)$pr_auc
      ),
    .id = "ensemble_type"
    )  |>
    mutate(y_pos = c(0.6, 0.8, 0.7, 0.9))

precision_data <- file_paths_precison  |>
  map_dfr(
    .f = ~ data.frame(
      precision = jsonlite::read_json(.x)$prec_value
      ),
    .id = "ensemble_type"
    )

recall_data <- file_paths_recall  |>
  map_dfr(
    .f = ~ data.frame(
      recall = jsonlite::read_json(.x)$rec_value
      ),
    .id = "ensemble_type"
    )

f1_data <- file_paths_f1  |>
  map_dfr(
    .f = ~ data.frame(
      f1 = jsonlite::read_json(.x)$f1_value
      ),
    .id = "ensemble_type"
    )

point_data <- precision_data  |>
  inner_join(recall_data, by = "ensemble_type")  |>
  inner_join(f1_data, by = "ensemble_type")  |>
  inner_join(pr_auc, by = "ensemble_type")  |>
  select(-y_pos)

manual_colours <-  RColorBrewer::brewer.pal(name = "Dark2", n = 4)

ensemle_types <- c(
  "top-20-ensemble",
  "one-model-all-prompts",
  "one-prompt-all-models",
  "one-model-one-prompt"
)

manual_labels <- c(
  "top-20-ensemble\n (AUC = 0.411)",
  "one-model-all-prompts\n (AUC = 0.344)",
  "one-prompt-all-models\n (AUC = 0.375)",
  "one-model-one-prompt \n (AUC = 0.235)"
)
manual_labels <- setNames(manual_labels, ensemle_types)
manual_colours <- setNames(manual_colours, ensemle_types)

manual_linetypes <- c(
  "one-model-all-prompts" = "dashed",
  "top-20-ensemble" = "solid",
  "one-prompt-all-models" = "dotted",
   "one-model-one-prompt" = "dotdash")

g <- ggplot(pr_curve_data, aes(x = recall, y = precision, color = ensemble_type, linetype = ensemble_type)) +
  geom_path(linejoin = "mitre") +
  scale_color_manual(
    values = manual_colours,
    labels = manual_labels) +
  scale_linetype_manual(
    values = manual_linetypes,
    labels = manual_labels) +
  geom_point(data = point_data, 
    mapping = aes(x = recall, y = precision, color =  ensemble_type),
    shape = 4,
    size =  0.5,
    stroke = 1.5) +
  # geom_label(
  #   data = pr_auc, 
  #   mapping = aes(x = 0.8, y = y_pos, color = ensemble_type,
  #                 label = paste("AUC = ", round(pr_auc,3))),
  #   size = 4,
  #   key_glyph = "blank"
  #   ) + 
  # geom_text_repel(
  #   data = f1_opt,
  #   mapping = aes(x = rec + 0.1, y = prec_cummax, color = family,
  #                 label = paste("f1 = ", round(f1_max,2)))
  # ) + 
  xlab("Recall") + 
  ylab("Precision") + 
  coord_fixed(xlim = c(0,1), ylim = c(0,1)) + 
  theme_minimal() + 
  guides(
    color = guide_legend(title = "Ensemble Strategy", shape = 1, ncol = 2, title.position = "top"),
    linetype = guide_legend(title = "Ensemble Strategy", ncol = 2)
  ) + 
  theme(legend.position = "bottom") 

g
ggsave(plot = g, 
  filename = "reports/pr_curves.png", 
  width = 10, 
  height = 12, 
  dpi = 300, 
  scale = 0.4, bg = "white")

knitr::kable(point_data)

knitr::kable(point_data, digits = 3, col.names = c("Ensemble Strategy", "Precision", "Recall", "F1", "PR-AUC"), format = "markdown")
