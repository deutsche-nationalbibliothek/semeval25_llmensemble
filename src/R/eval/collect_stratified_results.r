suppressPackageStartupMessages({
  library(tidyverse)
  library(arrow)
  library(optparse)
})

option_list <- list(
  make_option(
    c("--output"),
    type = "character", default = "out.arrow",
    help = "path to the output folder",
    metavar = "character"
  )
)
opt_parser <- OptionParser(option_list = option_list)
arguments <- parse_args(opt_parser, positional_arguments = TRUE)
opt <- arguments$options
args <- arguments$args

in_files <- args

n_in_files <- length(in_files)

read_file <- function(in_file) {
  path_components <- str_split(in_file, "/") |>
    unlist()
  l <- length(path_components)
  model_name <- path_components[l - 2]
  promptname <- path_components[l - 1]
  data <- read_feather(in_file)  |>
    mutate(
      model = model_name,
      prompt = promptname
    )
  return(data)
}

collected_results <- map_dfr(
  in_files, read_file
)

collected_results |>
  write_feather(opt$output)
