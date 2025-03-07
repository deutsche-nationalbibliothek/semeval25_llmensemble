# File name: analyse_single_model_results.r
# Description: Plots the stratified results by different models and prompts

library(tidyverse)
library(arrow)

df <- read_feather("results/stratified_results_by_model_and_prompt.arrow")

plot_data <- df  |>
  filter(text_type == "all")  |>
  filter(language == "all")  |>
  select(model, prompt, metric, value)  |>
  pivot_wider(names_from = metric, values_from = value)  |>
  arrange(desc(f1))

labels <- c(
  "llama-32-3B" = "Llama-3.2-3B-Instruct",
  "mistral-7B" = "Mistral-7B-v0.1",
  "mistral-7B-0p3" = "Mistral-7B-Instruct-v0.3",
  "openhermes-2p5-7B" = "OpenHermes-2.5-Mistral-7B",
  "teuken-7B-0p4" = "Teuken-7B-instruct-research-v0.4",
  "llama31-70B" = "Meta-Llama-3.1-70B-Instruct",
  "mistral-8x7B-0p1" = "Mixtral-8x7B-Instruct-v0.1"
)

h <- ggplot(
  data = plot_data,
  aes(x = rec, y = prec, color = model, shape = model)
) +
  geom_point() +
  scale_shape_manual(values = 1:7, labels = labels) +
  scale_color_brewer(palette = "Set1", labels = labels) +
  coord_fixed(xlim = c(0, 0.55), ylim = c(0, 0.55)) +
  labs(x = "Recall", y = "Precision") +
  guides(
    color = guide_legend(title = "LLM", ncol = 2, title.position = "top"),
    shape = guide_legend(title = "LLM", ncol = 2, title.position = "top")
  ) +
  theme_minimal() + 
  theme(legend.position = "bottom")

ggsave(plot = h, 
  filename = "reports/single_model_metrics.png", 
  width = 10, 
  height = 12, 
  dpi = 300, 
  scale = 0.48, bg = "white")
